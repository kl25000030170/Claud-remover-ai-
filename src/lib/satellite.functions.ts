import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

const inputSchema = z.object({
  imageBase64: z.string().min(10),
  mediaType: z.string().default("image/jpeg"),
});

export const analyzeSatelliteImage = createServerFn({ method: "POST" })
  .inputValidator((data: unknown) => inputSchema.parse(data))
  .handler(async ({ data }) => {
    const apiKey = process.env.LOVABLE_API_KEY;
    if (!apiKey) {
      throw new Error("AI gateway is not configured");
    }

    const prompt = `You are a satellite image analysis expert. Analyze this image and respond ONLY with a JSON object. No markdown, no explanation, just raw JSON.

Respond with exactly this structure:
{
  "isSatelliteImage": boolean,
  "satelliteConfidence": number,
  "cloudPercentage": number,
  "hasHighDensityClouds": boolean,
  "cloudRegions": [
    { "xPercent": number, "yPercent": number, "widthPercent": number, "heightPercent": number, "density": number, "shape": "circular"|"irregular"|"linear"|"patchy" }
  ],
  "cloudThresholds": { "brightnessMin": number, "saturationMax": number },
  "terrainFeatures": string[],
  "terrainContext": {
    "primaryLandUse": string,
    "typicalColorR": number,
    "typicalColorG": number,
    "typicalColorB": number,
    "textureComplexity": "low"|"medium"|"high"
  },
  "notSatelliteReason": string
}

Rules:
- isSatelliteImage = true only if this is a top-down aerial/satellite view of Earth's surface
- cloudRegions must reflect actual cloud positions in THIS specific image
- cloudThresholds must be calibrated to THIS image's actual brightness values (0-255)
- terrainFeatures from: roads, buildings, vegetation, agriculture, water, forest, urban, mountains, desert, coastline, farmland, river
- typicalColorR/G/B are 0-255 RGB values of the dominant ground terrain
- If not a satellite image, set isSatelliteImage=false and explain in notSatelliteReason`;

    const res = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              {
                type: "image_url",
                image_url: { url: `data:${data.mediaType};base64,${data.imageBase64}` },
              },
            ],
          },
        ],
      }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`AI gateway error ${res.status}: ${text.slice(0, 200)}`);
    }

    const payload = await res.json();
    const text: string = payload?.choices?.[0]?.message?.content ?? "";
    const clean = text.replace(/```json|```/g, "").trim();

    // Try to find a JSON object substring
    let parsed: any;
    try {
      parsed = JSON.parse(clean);
    } catch {
      const match = clean.match(/\{[\s\S]*\}/);
      if (!match) throw new Error("AI analysis failed — please try a clearer satellite image");
      parsed = JSON.parse(match[0]);
    }

    return parsed as {
      isSatelliteImage: boolean;
      satelliteConfidence: number;
      cloudPercentage: number;
      hasHighDensityClouds: boolean;
      cloudRegions: Array<{
        xPercent: number;
        yPercent: number;
        widthPercent: number;
        heightPercent: number;
        density: number;
        shape: string;
      }>;
      cloudThresholds: { brightnessMin: number; saturationMax: number };
      terrainFeatures: string[];
      terrainContext: {
        primaryLandUse: string;
        typicalColorR: number;
        typicalColorG: number;
        typicalColorB: number;
        textureComplexity: string;
      };
      notSatelliteReason: string | null;
    };
  });
