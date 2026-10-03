require("dotenv").config();

const express = require("express");
const cors = require("cors");
const client = require("prom-client");
const analyzeRoute = require("./routes/analyze");

const app = express();
const PORT = process.env.PORT || 5000;

// ======================================================
// PROMETHEUS CONFIGURATION
// ======================================================

const register = new client.Registry();

// Collect default Node.js metrics
client.collectDefaultMetrics({
  register,
});

// Total HTTP requests
const httpRequestCounter = new client.Counter({
  name: "agrisense_http_requests_total",
  help: "Total number of HTTP requests received by Agrisense",
  labelNames: ["method", "route", "status_code"],
});

register.registerMetric(httpRequestCounter);

// HTTP request latency
const httpRequestDuration = new client.Histogram({
  name: "agrisense_http_request_duration_seconds",
  help: "HTTP request duration in seconds",
  labelNames: ["method", "route", "status_code"],
  buckets: [0.1, 0.3, 0.5, 1, 2, 5],
});

register.registerMetric(httpRequestDuration);

// Application availability
const applicationUp = new client.Gauge({
  name: "agrisense_application_up",
  help: "Agrisense application availability status",
});

applicationUp.set(1);

register.registerMetric(applicationUp);

// ======================================================
// MIDDLEWARE
// ======================================================

app.use(cors());
app.use(express.json());

// HTTP monitoring middleware
app.use((req, res, next) => {
  // Don't count Prometheus scraping itself as an application request
  if (req.path === "/metrics") {
    return next();
  }

  const start = process.hrtime();

  res.on("finish", () => {
    const diff = process.hrtime(start);
    const duration = diff[0] + diff[1] / 1e9;

    const route = req.route?.path || req.path;

    httpRequestCounter.inc({
      method: req.method,
      route,
      status_code: res.statusCode,
    });

    httpRequestDuration.observe(
      {
        method: req.method,
        route,
        status_code: res.statusCode,
      },
      duration
    );
  });

  next();
});

// ======================================================
// AGRISENSE ROUTES
// ======================================================

app.use("/api/analyze", analyzeRoute);

// Health check
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "agrisense-ai-server",
  });
});

// ======================================================
// PROMETHEUS METRICS ENDPOINT
// ======================================================

app.get("/metrics", async (req, res) => {
  try {
    res.set("Content-Type", register.contentType);
    res.end(await register.metrics());
  } catch (error) {
    res.status(500).end(error.message);
  }
});

// ======================================================
// START SERVER
// ======================================================

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});