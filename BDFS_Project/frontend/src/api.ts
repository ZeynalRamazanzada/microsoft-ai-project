import axios from "axios";

const client = axios.create({
  baseURL: "/api",
  timeout: 15000,
});

export interface SchemaField {
  name: string;
  type: "int" | "float" | string;
  min: number;
  max: number;
  example: number;
  group: string;
}

export interface SchemaResponse {
  fields: SchemaField[];
}

export interface FeatureContribution {
  feature: string;
  value: number;
}

export interface PredictResponse {
  model_used: string;
  predicted_class: number;
  label: "fatigued" | "not_fatigued";
  probability: number;
  probability_not_fatigued: number;
  top_contributing_features: FeatureContribution[];
}

export type FeatureValues = Record<string, number>;

export async function fetchSchema(): Promise<SchemaResponse> {
  const { data } = await client.get<SchemaResponse>("/schema");
  return data;
}

export async function fetchExample(
  label: "fatigued" | "not_fatigued"
): Promise<FeatureValues> {
  const { data } = await client.get<FeatureValues>(`/example/${label}`);
  return data;
}

export async function predict(features: FeatureValues): Promise<PredictResponse> {
  const { data } = await client.post<PredictResponse>("/predict", { features });
  return data;
}

export async function health(): Promise<{ status: string; models_loaded: string[] }> {
  const { data } = await client.get("/health");
  return data;
}
