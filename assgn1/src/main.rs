use std::env;
use std::path::PathBuf;

use ffmpeg_sidecar::command::FfmpegCommand;
use workspace_root::get_workspace_root;

use std::fs::File;
use std::io::BufReader;
use std::io::{BufWriter, Write};

use bitbit::BitReader;
use bitbit::BitWriter;
use bitbit::MSB;

use toy_ac::decoder::Decoder;
use toy_ac::encoder::Encoder;
use toy_ac::symbol_model::VectorCountSymbolModel;

use ffmpeg_sidecar::event::StreamTypeSpecificData::Video;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Make sure ffmpeg is installed
    ffmpeg_sidecar::download::auto_download().unwrap();

    // Command line options
    // -verbose, -no_verbose                Default: -no_verbose
    // -report, -no_report                  Default: -report
    // -check_decode, -no_check_decode      Default: -no_check_decode
    // -skip_count n                        Default: -skip_count 0
    // -count n                             Default: -count 10
    // -in file_path                        Default: bourne.mp4 in data subdirectory of workplace
    // -out file_path                       Default: out.dat in data subdirectory of workplace

    // Set up default values of options
    let mut verbose = false;
    let mut report = true;
    let mut check_decode = false;
    let mut skip_count = 0;
    let mut count = 10;

    let mut data_folder_path = get_workspace_root();
    data_folder_path.push("data");

    let mut input_file_path = data_folder_path.join("bourne.mp4");
    let mut output_file_path = data_folder_path.join("out.dat");

    parse_args(
        &mut verbose,
        &mut report,
        &mut check_decode,
        &mut skip_count,
        &mut count,
        &mut input_file_path,
        &mut output_file_path,
    );

    // Run an FFmpeg command to decode video from inptu_file_path
    // Get output as grayscale (i.e., just the Y plane)

    let mut iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
        .input(input_file_path.to_str().unwrap())
        .format("rawvideo")
        .pix_fmt("gray8")
        .output("-")
        .spawn()? // <- Ordinary `std::process::Child`
        .iter()?; // <- Blocking iterator over logs and output

    // Figure out geometry of frame.
    let mut width = 0;
    let mut height = 0;

    let metadata = iter.collect_metadata()?;
    for i in 0..metadata.output_streams.len() {
        match &metadata.output_streams[i].type_specific_data {
            Video(vid_stream) => {
                width = vid_stream.width;
                height = vid_stream.height;

                if verbose {
                    println!(
                        "Found video stream at output stream index {} with dimensions {} x {}",
                        i, width, height
                    );
                }
                break;
            }
            _ => (),
        }
    }
    assert!(width != 0);
    assert!(height != 0);

    // Set up initial prior frame as uniform medium gray (y = 128)
    let mut prior_frame = vec![128 as u8; (width * height) as usize];

    let output_file = match File::create(&output_file_path) {
        Err(_) => panic!("Error opening output file"),
        Ok(f) => f,
    };

    // Setup bit writer and arithmetic encoder.

    let mut buf_writer = BufWriter::new(output_file);
    let mut bw = BitWriter::new(&mut buf_writer);

    let mut enc = Encoder::new();

    // Two adaptive symbol models:
    //   intra_pdf  — used for frame 0 (no prior frame), predicts each pixel
    //                from its left neighbor in the current frame (spatial prediction).
    //   inter_pdf  — used for all subsequent frames, predicts each pixel from
    //                the same pixel in the prior frame (temporal prediction) refined
    //                by also subtracting the left neighbor's temporal difference
    //                (spatial-temporal prediction).
    let mut intra_pdf = VectorCountSymbolModel::new((0..=255).collect());
    let mut inter_pdf = VectorCountSymbolModel::new((0..=255).collect());

    let mut frame_index = 0u32; // counts frames actually encoded (after skipping)

    // Process frames
    for frame in iter.filter_frames() {
        if frame.frame_num < skip_count {
            if verbose {
                println!("Skipping frame {}", frame.frame_num);
            }
        } else if frame.frame_num < skip_count + count {
            let current_frame: Vec<u8> = frame.data; // <- raw pixel y values

            let bits_written_at_start = enc.bits_written();

            // Process pixels in row major order.
            for r in 0..height {
                for c in 0..width {
                    let pixel_index = (r * width + c) as usize;
                    let current_val = current_frame[pixel_index] as i32;

                    let residual = if frame_index == 0 {
                        // Intra frame: predict from left neighbor (or 128 for first column).
                        let left_val = if c == 0 {
                            128i32
                        } else {
                            current_frame[pixel_index - 1] as i32
                        };
                        let prediction = left_val;
                        ((current_val - prediction) + 256) % 256
                    } else {
                        // Inter frame: predict from the co-located pixel in the prior frame,
                        // then further refine using the left neighbor's temporal difference
                        // as a spatial correction (reduces residual entropy significantly).
                        let temporal_pred = prior_frame[pixel_index] as i32;
                        let left_temporal_diff = if c == 0 {
                            0i32
                        } else {
                            // difference already encoded for the pixel to our left this frame
                            current_frame[pixel_index - 1] as i32
                                - prior_frame[pixel_index - 1] as i32
                        };
                        let prediction = (temporal_pred + left_temporal_diff).clamp(0, 255);
                        ((current_val - prediction) + 256) % 256
                    };

                    if frame_index == 0 {
                        enc.encode(&residual, &intra_pdf, &mut bw);
                        intra_pdf.incr_count(&residual);
                    } else {
                        enc.encode(&residual, &inter_pdf, &mut bw);
                        inter_pdf.incr_count(&residual);
                    }
                }
            }

            prior_frame = current_frame;
            frame_index += 1;

            let bits_written_at_end = enc.bits_written();

            if verbose {
                println!(
                    "frame: {}, compressed size (bits): {}",
                    frame.frame_num,
                    bits_written_at_end - bits_written_at_start
                );
            }
        } else {
            break;
        }
    }

    // Tie off arithmetic encoder and flush to file.
    enc.finish(&mut bw)?;
    bw.pad_to_byte()?;
    buf_writer.flush()?;

    // Decompress and check for correctness.
    if check_decode {
        let output_file = match File::open(&output_file_path) {
            Err(_) => panic!("Error opening output file"),
            Ok(f) => f,
        };
        let mut buf_reader = BufReader::new(output_file);
        let mut br: BitReader<_, MSB> = BitReader::new(&mut buf_reader);

        let iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
            .input(input_file_path.to_str().unwrap())
            .format("rawvideo")
            .pix_fmt("gray8")
            .output("-")
            .spawn()? // <- Ordinary `std::process::Child`
            .iter()?; // <- Blocking iterator over logs and output

        let mut dec = Decoder::new();

        let mut dec_intra_pdf = VectorCountSymbolModel::new((0..=255).collect());
        let mut dec_inter_pdf = VectorCountSymbolModel::new((0..=255).collect());

        // Set up initial prior frame as uniform medium gray
        let mut prior_frame = vec![128 as u8; (width * height) as usize];
        let mut dec_frame_index = 0u32;

        'outer_loop: 
        for frame in iter.filter_frames() {
            if frame.frame_num < skip_count + count {
                if verbose {
                    print!("Checking frame: {} ... ", frame.frame_num);
                }

                let current_frame: Vec<u8> = frame.data; // <- raw pixel y values
                let mut reconstructed = vec![0u8; (width * height) as usize];

                // Process pixels in row major order.
                for r in 0..height {
                    for c in 0..width {
                        let pixel_index = (r * width + c) as usize;

                        let residual = if dec_frame_index == 0 {
                            let decoded = dec.decode(&dec_intra_pdf, &mut br).to_owned();
                            dec_intra_pdf.incr_count(&decoded);
                            decoded
                        } else {
                            let decoded = dec.decode(&dec_inter_pdf, &mut br).to_owned();
                            dec_inter_pdf.incr_count(&decoded);
                            decoded
                        };

                        let prediction = if dec_frame_index == 0 {
                            let left_val = if c == 0 {
                                128i32
                            } else {
                                reconstructed[pixel_index - 1] as i32
                            };
                            left_val
                        } else {
                            let temporal_pred = prior_frame[pixel_index] as i32;
                            let left_temporal_diff = if c == 0 {
                                0i32
                            } else {
                                reconstructed[pixel_index - 1] as i32
                                    - prior_frame[pixel_index - 1] as i32
                            };
                            (temporal_pred + left_temporal_diff).clamp(0, 255)
                        };

                        let pixel_value = (prediction + residual) % 256;
                        reconstructed[pixel_index] = pixel_value as u8;

                        if pixel_value != current_frame[pixel_index] as i32 {
                            println!(
                                " error at ({}, {}), should decode {}, got {}",
                                c, r, current_frame[pixel_index], pixel_value
                            );
                            println!("Abandoning check of remaining frames");
                            break 'outer_loop;
                        }
                    }
                }
                println!("correct.");
                prior_frame = reconstructed;
                dec_frame_index += 1;
            } else {
                break 'outer_loop;
            }
        }
    }

    // Emit report
    if report {
        println!(
            "{} frames encoded, average size (bits): {}, compression ratio: {:.2}",
            count,
            enc.bits_written() / count as u64,
            (width * height * 8 * count) as f64 / enc.bits_written() as f64
        )
    }

    Ok(())
}

fn parse_args(
    verbose: &mut bool,
    report: &mut bool,
    check_decode: &mut bool,
    skip_count: &mut u32,
    count: &mut u32,
    input_file_path: &mut PathBuf,
    output_file_path: &mut PathBuf,
) -> () {
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "-verbose" {
            *verbose = true;
        } else if arg == "-no_verbose" {
            *verbose = false;
        } else if arg == "-report" {
            *report = true;
        } else if arg == "-no_report" {
            *report = false;
        } else if arg == "-check_decode" {
            *check_decode = true;
        } else if arg == "-no_check_decode" {
            *check_decode = false;
        } else if arg == "-skip_count" {
            match args.next() {
                Some(skip_count_string) => {
                    *skip_count = skip_count_string.parse::<u32>().unwrap();
                }
                None => {
                    panic!("Expected count after -skip_count option");
                }
            }
        } else if arg == "-count" {
            match args.next() {
                Some(count_string) => {
                    *count = count_string.parse::<u32>().unwrap();
                }
                None => {
                    panic!("Expected count after -count option");
                }
            }
        } else if arg == "-in" {
            match args.next() {
                Some(input_file_path_string) => {
                    *input_file_path = PathBuf::from(input_file_path_string);
                }
                None => {
                    panic!("Expected input file name after -in option");
                }
            }
        } else if arg == "-out" {
            match args.next() {
                Some(output_file_path_string) => {
                    *output_file_path = PathBuf::from(output_file_path_string);
                }
                None => {
                    panic!("Expected output file name after -out option");
                }
            }
        }
    }
}
