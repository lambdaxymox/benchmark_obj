pub mod lexer;
pub mod lexer2;
pub mod parser1;
pub mod parser2;

use crate::lexer::Lexer;
use std::iter;
use std::error;
use std::fmt;
use std::io::{BufReader, Read};
use std::fs::File;
use std::path::Path;


const SAMPLE_DATA: &str = "assets/teapot.obj";


fn main() {
    println!("BEGING PARSING.");
    let result: parser2::Object = parser2::parse_file(SAMPLE_DATA).unwrap();
    println!("END PARSING.");
}
