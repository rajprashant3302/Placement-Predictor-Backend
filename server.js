const express = require("express");
const { spawn } = require("child_process");
const path = require("path");
const cors = require("cors");

const app = express();
app.use(express.json());
app.use(cors({
  origin: ["http://localhost:5173", "https://placement-predictor-frontend-three.vercel.app"],
}));

app.post("/predict", (req, res) => {

  // console.log("Received req:", req);
  const features = req.body;

  // console.log("Received features:", features);

  const python = spawn("python", [path.join(__dirname, "model.py"),JSON.stringify(features)]);

  let result = "";
  let error = "";


  python.stdout.on("data", (data) => {
    result += data.toString();
  });

  python.stderr.on("data", (data) => {
    error += data.toString();
  });

  python.on("close", (code) => {
    if (error) {
      // console.error("Python error:", error);
      return res.status(500).json({ error });
    }
    // console.log("Prediction:", result.trim());
    res.json({ prediction: result.trim() });
  });
});

app.listen(5000, () => console.log("✅ Backend running on http://localhost:5000"));
