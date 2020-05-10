extern crate wavefront_obj;


const SAMPLE_DATA: &str = "assets/teapot.obj";


fn main() {
    println!("BEGING PARSING.");
    wavefront_obj::parse_file(SAMPLE_DATA).unwrap();
    println!("END PARSING.");
}
