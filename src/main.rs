pub mod lexer;
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
    let result = parser1::parse_file(SAMPLE_DATA);
    println!("END PARSING.");
    /*
    let file = File::open(SAMPLE_DATA).unwrap();
    let mut reader = BufReader::new(file);
    let mut string = String::new();
    reader.read_to_string(&mut string).unwrap();
    let lexer = Lexer::new(string.chars());
    for token in lexer {
        // Run through the lexer.
    }
    println!("LEXING DONE.");
    */
}
