// app/page.tsx
"use client";

import { useState } from "react";
import Logo from "@app/assets/icon-logo.svg";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<null | {
    Prediction: "REAL" | "FAKE";
    FeatureDistributionPlotURL: string;
    FeatureImportancePlotURL: string;
    FeatureStatsPlotURL: string;
    [key: string]: any;
  }>(null);

  const [showModal, setShowModal] = useState(false);
  const [modalImage, setModalImage] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0];
    if (!uploadedFile) return;

    if (uploadedFile.type !== "audio/wav") {
      setError("Only .wav files are allowed");
      setFile(null);
    } else {
      setError(null);
      setFile(uploadedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setError("Please upload a .wav file before submitting.");
      return;
    }

    const formData = new FormData();
    formData.append("audio_file", file);

    try {
      setLoading(true);
      setResult(null);
      const res = await fetch("http://127.0.0.1:5000/verify-audio", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to verify audio.");

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-[70px]  p-20">
      <div className=" flex items-center gap-3">
        <Logo className="scale-[0.7]" />

        <div className="flex flex-col text-white gap-5">
          <h1 className=" text-7xl font-semibold">SafeSpeak</h1>
          <h2 className=" text-3xl font-semibold tracking-[15px]">
            SAY IT SAFE
          </h2>
        </div>
      </div>

      <div className="flex flex-col w-[500px] justify-center items-center mx-auto">
        <h1 className="text-3xl font-semibold mb-4 text-center text-white">
          Upload a WAV File
        </h1>

        <form
          onSubmit={handleSubmit}
          className="bg-gradient-to-b from-[#0064BB] to-[#0081CC] border-[1.5px] border-white/50 rounded-2xl px-16 py-15 w-full"
        >
          <input
            type="file"
            accept=".wav"
            onChange={handleFileChange}
            className="file:mr-4 file:rounded-xl file:border-0 file:bg-white file:px-4 file:py-1 text-[18px] file:font-semibold file:text-[#0064BB] hover:file:bg-white/90 text-white"
          />

          {error && <p className="text-white my-5 text-lg">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className=" w-full bg-gradient-to-r from-[#1AC4EF] to-[#10AAE3] hover:from-[#1AC4EF]/80 hover:to-[#10AAE3]/80 text-white py-2 px-4 rounded-xl text-[18px] font-semibold disabled:opacity-50"
          >
            {loading ? "Processing..." : "Submit"}
          </button>
        </form>
      </div>

      {result && (
        <div className="mt-8 p-10 bg-white rounded-2xl shadow-md w-full max-w-4xl text-center">
          <h2 className="text-xl font-bold">
            Prediction:{" "}
            <span
              className={
                result.Prediction === "REAL" ? "text-green-600" : "text-red-600"
              }
            >
              {result.Prediction}
            </span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {result.FeatureDistributionPlotURL && (
              <div>
                <h3 className="text-lg font-semibold mt-4">Model Plots</h3>
                <div
                  className="cursor-pointer"
                  onClick={() => {
                    setModalImage(result.FeatureDistributionPlotURL);
                    setShowModal(true);
                  }}
                >
                  <img
                    src={result.FeatureDistributionPlotURL}
                    alt="Model Plots"
                    className="rounded-md"
                  />
                </div>
              </div>
            )}
            {result.FeatureImportancePlotURL && (
              <div>
                <h3 className="text-lg font-semibold mt-4">
                  Feature Importance
                </h3>
                <div
                  className="cursor-pointer"
                  onClick={() => {
                    setModalImage(result.FeatureImportancePlotURL);
                    setShowModal(true);
                  }}
                >
                  <img
                    src={result.FeatureImportancePlotURL}
                    alt="Feature Importance"
                    className="rounded-md"
                  />
                </div>
              </div>
            )}
            {result.FeatureStatsPlotURL && (
              <div>
                <h3 className="text-lg font-semibold mt-4">Feature Stats</h3>
                <div
                  className="cursor-pointer"
                  onClick={() => {
                    setModalImage(result.FeatureStatsPlotURL);
                    setShowModal(true);
                  }}
                >
                  <img
                    src={result.FeatureStatsPlotURL}
                    alt="Feature Stats"
                    className="rounded-md"
                  />
                </div>
              </div>
            )}
          </div>
          {/* Display any other returned data */}
          {/*{Object.entries(result).map(([key, val]) =>
            key !== "Prediction" && key !== "spectrogramUrl" ? (
              <p key={key} className="text-sm mt-2">
                <strong>{key}:</strong> {String(val)}
              </p>
            ) : null
          )}*/}

          {/* Modal for enlarged image */}
          {showModal && modalImage && (
            <div
              className="fixed inset-0 bg-black bg-opacity-80 flex justify-center items-center z-50"
              onClick={() => setShowModal(false)}
            >
              <img
                src={modalImage}
                alt="Enlarged"
                className="max-w-[90%] max-h-[90%] rounded-lg boarder-4 boarder-white"
                onClick={(e) => {
                  e.stopPropagation();
                }}
              />
            </div>
          )}
        </div>
      )}
    </main>
  );
}
