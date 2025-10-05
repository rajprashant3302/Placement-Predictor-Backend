// backend/server.js
const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const path = require("path");
const cors = require("cors"); // ✅ import CORS

const app = express();

// Middleware
app.use(bodyParser.json());

// Enable CORS for your frontend port
app.use(cors({
  origin: "https://placement-predictor-frontend-three.vercel.app", // React dev server
}));

// Endpoint: POST /predict
app.post("/predict", (req, res) => {
  const features = req.body.features; // Example: [5.1, 3.5, 1.4, 0.2]

  // Spawn Python process to run model.py
  const python = spawn("python", [
    path.join(__dirname, "model.py"),
    JSON.stringify(features),
  ]);

  let result = "";

  python.stdout.on("data", (data) => {
    result += data.toString();
  });

  python.stderr.on("data", (data) => {
    console.error(`Python error: ${data}`);
  });

  python.on("close", () => {
    res.json({ prediction: result.trim() });
  });
});

// Start server
const PORT = 5000;
app.listen(PORT, () => console.log(`✅ Backend running on http://localhost:${PORT}`));


