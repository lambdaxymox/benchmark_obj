use criterion::{
    black_box, 
    criterion_group, 
    criterion_main, 
    Criterion
};
use std::fs::File;
use std::io::{
    BufReader,
    Read
};

use wavefront_obj as piston_obj;

const SAMPLE_DATA: &str = "assets/teapot.obj";


fn benchmark_piston(c: &mut Criterion) {
    c.bench_function("piston parser teapot.obj", |b| b.iter(|| {
        let file = File::open(SAMPLE_DATA).unwrap();
        let mut reader = BufReader::new(file);
        let mut string = String::new();
        reader.read_to_string(&mut string).unwrap();
        let result = piston_obj::obj::parse(black_box(string));
        result.unwrap()
    }));
}

criterion_group!(benches, benchmark_piston);
criterion_main!(benches);
