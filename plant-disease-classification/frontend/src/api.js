import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function fetchHistory(limit = 20) {
  const res = await api.get("/history", { params: { limit } });
  return res.data;
}

export async function fetchHistoryItem(id) {
  const res = await api.get(`/history/${id}`);
  return res.data;
}

export async function deleteHistoryItem(id) {
  await api.delete(`/history/${id}`);
}
