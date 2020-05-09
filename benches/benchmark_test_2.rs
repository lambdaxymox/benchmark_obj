use criterion::{
    black_box, 
    criterion_group, 
    criterion_main, 
    Criterion
};
use benchmark_obj as obj;

const SAMPLE_DATA: &str = "assets/test.obj";


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("parser2 test.obj", |b| b.iter(|| {
        let result = obj::parser2::parse_file(black_box(SAMPLE_DATA));
        result.unwrap()
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
