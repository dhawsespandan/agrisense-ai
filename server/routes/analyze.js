const express = require("express");
const router = express.Router();
const axios = require("axios");
const FormData = require("form-data");
const upload = require("../middleware/upload");

router.post("/", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image provided" });
    }

    // Forward image to Python FastAPI service
    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const pythonRes = await axios.post(
      `${process.env.PYTHON_SERVICE_URL}/predict`,
      form,
      { headers: form.getHeaders() }
    );

    return res.json(pythonRes.data);
  } catch (err) {
    console.error("Analysis error:", err.message);
    return res.status(500).json({ error: "Analysis failed", detail: err.message });
  }
});

module.exports = router;