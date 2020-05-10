pub mod lexer;
pub mod lexer2;
pub mod parser1;
pub mod parser2;


const SAMPLE_DATA: &str = "assets/teapot.obj";


fn main() {
    println!("BEGING PARSING.");
    parser2::parse_file(SAMPLE_DATA).unwrap();
    println!("END PARSING.");
}
