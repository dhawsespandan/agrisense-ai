require("dotenv").config();
const express = require("express");
const cors = require("cors");
const analyzeRoute = require("./routes/analyze");

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Routes
app.use("/api/analyze", analyzeRoute);

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "agrisense-ai-server" });
});

// IMPORTANT FIX
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});