import type { DetectionResult } from "@/types";

export const NAV_ITEMS = ["Detection", "History", "Reports"] as const;

export const CAPABILITY_TAGS = [
  "Fruit & Leaf Disease Detection",
  "Disease Severity Estimation",
  "Flower Cluster Recognition",
] as const;

export const MOCK_RESULTS: DetectionResult[] = [
  {
    disease: "Apple Scab (Venturia inaequalis)",
    severity: "Moderate — 35% leaf area affected",
    confidence: "94.2%",
    recommendation:
      "Apply fungicide (Captan or Mancozeb) within 48 hours. Remove and dispose of affected leaves. Improve air circulation around the canopy.",
    details:
      "Apple scab is a common fungal disease causing dark, scab-like lesions on leaves and fruit. Early intervention is critical to prevent spread during wet conditions.",
  },
  {
    disease: "Powdery Mildew (Podosphaera leucotricha)",
    severity: "Mild — 12% leaf area affected",
    confidence: "91.7%",
    recommendation:
      "Spray with sulfur-based fungicide. Prune affected shoots. Avoid overhead irrigation to reduce humidity.",
    details:
      "Powdery mildew appears as a white powdery coating on young leaves and shoots. Warm days and cool nights favor its development.",
  },
  {
    disease: "Healthy Apple Leaf",
    severity: "No disease detected",
    confidence: "98.1%",
    recommendation:
      "No immediate action required. Continue regular monitoring every 7–10 days. Maintain optimal irrigation and nutrition schedule.",
    details:
      "The leaf shows no signs of fungal, bacterial, or pest damage. Keep monitoring as part of your integrated pest management routine.",
  },
];