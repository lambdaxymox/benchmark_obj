use criterion::{
    black_box, 
    criterion_group, 
    criterion_main, 
    Criterion
};
use benchmark_obj as obj;
use obj::lexer::{
    Lexer,
};
use obj::lexer2;
use std::fs::File;
use std::io::{
    BufReader,
    Read
};

use wavefront_obj as piston_obj;

const SAMPLE_DATA: &str = "assets/teapot.obj";


fn benchmark(c: &mut Criterion) {
    c.bench_function("parser2 teapot.obj", |b| b.iter(|| {
        let result = obj::parser2::parse_file(black_box(SAMPLE_DATA));
        result.unwrap()
    }));
}

fn benchmark_lexer(c: &mut Criterion) {
    c.bench_function("lexer1 teapot.obj", |b| b.iter(|| {
        let file = File::open(SAMPLE_DATA).unwrap();
        let mut reader = BufReader::new(file);
        let mut string = String::new();
        reader.read_to_string(&mut string).unwrap();
        let lexer = Lexer::new(string.chars());
        for token in lexer {
            // Run through the lexer.
        }
    }));
}

fn benchmark_lexer2(c: &mut Criterion) {
    c.bench_function("lexer2 teapot.obj", |b| b.iter(|| {
        let file = File::open(SAMPLE_DATA).unwrap();
        let mut reader = BufReader::new(file);
        let mut string = String::new();
        reader.read_to_string(&mut string).unwrap();
        let lexer = lexer2::Lexer::new(string.chars());
        for token in lexer {
            // Run through the lexer.
        }
    }));
}

criterion_group!(benches, benchmark, benchmark_lexer, benchmark_lexer2);
criterion_main!(benches);
