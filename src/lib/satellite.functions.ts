import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import fs from "node:fs";
import path from "node:path";
import { exec } from "node:child_process";
import { promisify } from "node:util";

const execPromise = promisify(exec);

const inputSchema = z.object({
  imageBase64: z.string().min(10),
  mediaType: z.string().default("image/jpeg"),
});

export const analyzeSatelliteImage = createServerFn({ method: "POST" })
  .inputValidator((data: unknown) => inputSchema.parse(data))
  .handler(async ({ data }) => {
    const timestamp = Date.now();

    // 1. FAST PATH: Attempt fetching from the persistent Python server
    const serverUrl = "http://127.0.0.1:8000/analyze";
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 seconds timeout

    try {
      console.log(`[DevOps] Attempting fast execution via python daemon at: ${serverUrl}`);
      const response = await fetch(serverUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageBase64: data.imageBase64,
          mediaType: data.mediaType,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const result = await response.json();
        if (result.error) {
          throw new Error(result.error);
        }
        console.log("[DevOps] Fast execution successful! No process spawned.");
        return result;
      } else {
        console.warn("[DevOps] Python server responded with error. Falling back to CLI execution.");
      }
    } catch (fetchErr) {
      clearTimeout(timeoutId);
      console.warn(`[DevOps] Python daemon not available or timed out: ${(fetchErr as Error).message}. Falling back to CLI execution.`);
    }

    // 2. SLOW FALLBACK: Standard process execution (files on disk)
    const tempDir = path.join(process.cwd(), "tmp_reconstruct_" + timestamp);

    // Ensure directory exists
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    let inputExt = "jpg";
    if (data.mediaType.includes("png")) {
      inputExt = "png";
    } else if (data.mediaType.includes("tiff") || data.mediaType.includes("tif")) {
      inputExt = "tiff";
    }
    const inputPath = path.join(tempDir, `input.${inputExt}`);

    // Write input image to disk
    const buffer = Buffer.from(data.imageBase64, "base64");
    fs.writeFileSync(inputPath, buffer);

    try {
      const pythonPath = path.join(process.cwd(), ".venv", "bin", "python");
      const scriptPath = path.join(process.cwd(), "src", "lib", "reconstruct.py");

      const cmd = `"${pythonPath}" "${scriptPath}" --image "${inputPath}" --out_dir "${tempDir}"`;

      const { stdout } = await execPromise(cmd);

      // Parse output JSON from Python stdout
      const result = JSON.parse(stdout.trim());

      if (result.isSatelliteImage === false) {
        // Clean up directory
        fs.rmSync(tempDir, { recursive: true, force: true });
        throw new Error(result.notSatelliteReason || "NOT_SATELLITE");
      }

      // Read outputs back and convert to base64
      const maskBuffer = fs.readFileSync(result.maskPath);
      const reconstBuffer = fs.readFileSync(result.reconstPath);
      const confidenceBuffer = fs.readFileSync(result.confidencePath);
      const terrainBuffer = fs.readFileSync(result.terrainMapPath);
      const originalPath = path.join(tempDir, "original.png");
      const originalBuffer = fs.readFileSync(originalPath);
      const overlayPath = path.join(tempDir, "cloud_overlay.png");
      const overlayBuffer = fs.readFileSync(overlayPath);

      const maskBase64 = `data:image/png;base64,${maskBuffer.toString("base64")}`;
      const reconstBase64 = `data:image/png;base64,${reconstBuffer.toString("base64")}`;
      const confidenceBase64 = `data:image/png;base64,${confidenceBuffer.toString("base64")}`;
      const terrainBase64 = `data:image/png;base64,${terrainBuffer.toString("base64")}`;
      const originalBase64 = `data:image/png;base64,${originalBuffer.toString("base64")}`;
      const overlayBase64 = `data:image/png;base64,${overlayBuffer.toString("base64")}`;

      // Clean up directory
      fs.rmSync(tempDir, { recursive: true, force: true });

      return {
        isSatelliteImage: result.isSatelliteImage,
        satelliteConfidence: result.satelliteConfidence,
        cloudPercentage: result.cloudPercentage,
        reconstructionConfidence: result.reconstructionConfidence,
        hasHighDensityClouds: result.cloudPercentage > 40,
        cloudRegions: [], // Handled entirely server-side now
        cloudThresholds: { brightnessMin: 0, saturationMax: 0 },
        terrainFeatures: result.terrainFeatures,
        terrainContext: {
          primaryLandUse: result.primaryLandUse,
          typicalColorR: result.typicalColorR,
          typicalColorG: result.typicalColorG,
          typicalColorB: result.typicalColorB,
          textureComplexity: result.textureComplexity,
        },
        notSatelliteReason: result.notSatelliteReason,
        isDemoMode: false,

        // Upgraded output images & AI validation metrics
        originalImage: originalBase64,
        reconstructedImage: reconstBase64,
        cloudMask: maskBase64,
        cloudOverlay: overlayBase64,
        confidenceMap: confidenceBase64,
        terrainClassificationMap: terrainBase64,
        psnr: result.psnr,
        ssim: result.ssim,
        processingTimeMs: result.processingTimeMs,
        inferenceTimeMs: result.inferenceTimeMs || 0,
        modelVersion: result.modelVersion || "unknown",
        qualityReport: result.qualityReport,
        predictedFeatures: result.predictedFeatures || [],

        // Phase 8 final analysis report fields
        cloud_percentage: result.cloud_percentage,
        inference_time_ms: result.inference_time_ms,
        total_processing_ms: result.total_processing_ms,
        psnr_db: result.psnr_db,
        ssim_score: result.ssim_score,
        reconstruction_confidence: result.reconstruction_confidence,
        terrain_prediction: result.terrain_prediction,
        terrain_confidence: result.terrain_confidence,
        model_name: result.model_name,
        model_version: result.model_version,
        device: result.device,
        timestamp: result.timestamp,
        reconstruction_note: result.reconstruction_note,
      };
    } catch (err) {
      // Cleanup on error
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
      throw new Error(`AI reconstruction failed: ${(err as Error).message}`);
    }
  });
