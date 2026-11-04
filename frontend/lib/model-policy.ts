export function modelAllowed(id: string): boolean {
  const model = id.replace(/^models\//, "");
  if (model === "gemini-3.5-flash-lite") return true;
  return !(
    /^gemini-3\.(5|6|7|8)-flash(?:$|-)/.test(model) ||
    /^(gemini-1\.|gemini-2\.0-)/.test(model) ||
    ["gemini-3-pro-preview", "gemini-3.1-flash-lite-preview"].includes(model)
  );
}
