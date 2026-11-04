import { expect, it } from "vitest";
import { modelAllowed } from "./model-policy";

it("excludes removed Flash families and their aliases, leaving other models intact", () => {
  for (const id of [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "models/gemini-3.8-flash-preview",
    "gemini-3.6-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite-preview",
  ])
    expect(modelAllowed(id)).toBe(false);
  for (const id of [
    "auto",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
  ])
    expect(modelAllowed(id)).toBe(true);
});
