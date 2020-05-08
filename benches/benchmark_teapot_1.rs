use criterion::{
    black_box, 
    criterion_group, 
    criterion_main, 
    Criterion
};
use wavefront_exp as obj;
use obj::lexer::{
    Lexer,
};
use std::fs::File;
use std::io::{
    BufReader,
    Read
};

const SAMPLE_DATA: &str = "assets/teapot.obj";


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("parser1 teapot.obj", |b| b.iter(|| {
        let result = obj::parser1::parse_file(black_box(SAMPLE_DATA));
        result.unwrap()
    }));
}

fn criterion_benchmark_lex(c: &mut Criterion) {
    c.bench_function("lexer teapot.obj", |b| b.iter(|| {
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

criterion_group!(benches, criterion_benchmark, criterion_benchmark_lex);
criterion_main!(benches);

