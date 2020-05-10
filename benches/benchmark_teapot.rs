use criterion::{
    black_box, 
    criterion_group, 
    criterion_main, 
    Criterion,
};

const SAMPLE_DATA: &str = "assets/teapot.obj";


fn benchmark_parser(c: &mut Criterion) {
    c.bench_function("parser teapot.obj", |b| b.iter(|| {
        let result = wavefront_obj::parse_file(black_box(SAMPLE_DATA));
        result.unwrap()
    }));
}

criterion_group!(benches, benchmark_parser);
criterion_main!(benches);
