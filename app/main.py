

from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

app = FastAPI()

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Setup OpenTelemetry Tracing
resource = Resource(attributes={
    "service.name": "embedding-service"
})

trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(_name_)

# Export spans to the OTEL Collector
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument FastAPI app and requests
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

@app.post("/embed")
async def embed_text(req: Request):
    data = await req.json()
    text = data.get("text", "")
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}