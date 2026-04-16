use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::time::Instant;

pub fn install() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder")
}

pub fn request_started(streaming: bool) {
    let mode = if streaming { "streaming" } else { "batch" };
    counter!("proxy_requests_total", "mode" => mode).increment(1);
    gauge!("proxy_requests_in_flight").increment(1.0);
}

pub fn request_finished(start: Instant, status: u16, streaming: bool) {
    let mode = if streaming { "streaming" } else { "batch" };
    let status_str = status.to_string();

    histogram!("proxy_request_duration_seconds", "mode" => mode, "status" => status_str.clone())
        .record(start.elapsed().as_secs_f64());
    counter!("proxy_responses_total", "mode" => mode, "status" => status_str).increment(1);
    gauge!("proxy_requests_in_flight").decrement(1.0);
}

pub fn upstream_latency(seconds: f64, endpoint: &'static str) {
    histogram!("proxy_upstream_latency_seconds", "endpoint" => endpoint).record(seconds);
}

pub fn tokens(input: u32, output: u32, model: &str) {
    let model = model.to_string();
    counter!("proxy_tokens_total", "type" => "input", "model" => model.clone())
        .increment(input as u64);
    counter!("proxy_tokens_total", "type" => "output", "model" => model).increment(output as u64);
}

pub fn upstream_error(endpoint: &'static str) {
    counter!("proxy_upstream_errors_total", "endpoint" => endpoint).increment(1);
}
