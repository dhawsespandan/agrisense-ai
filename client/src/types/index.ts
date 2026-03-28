export type ResultStatus = "idle" | "loading" | "success";

export interface DetectionResult {
  disease: string;
  severity: string;
  confidence: string;
  recommendation: string;
  details: string;
}