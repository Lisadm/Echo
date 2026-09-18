import { listen } from "@tauri-apps/api/event";
import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import "./RecordingOverlay.css";
import { commands } from "@/bindings";
import i18n, { syncLanguageFromSettings } from "@/i18n";
import { getLanguageDirection } from "@/lib/utils/rtl";

type OverlayState =
  | "preparing"
  | "recording"
  | "transcribing"
  | "processing"
  // Wake listening (plan step 13): the user must always see whether the mic
  // is armed, hearing speech, or waiting for the follow-up command.
  | "armed"
  | "wake-listening"
  | "wake-activated";

const SUBTITLE_FONT_PX: Record<string, number> = {
  small: 14,
  medium: 18,
  large: 24,
};

const RecordingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const [state, setState] = useState<OverlayState>("preparing");
  const [level, setLevel] = useState(0);
  const [subtitles, setSubtitles] = useState("");
  const smoothedLevel = useRef(0);
  const direction = getLanguageDirection(i18n.language);

  useEffect(() => {
    const setup = async () => {
      const unlistenShow = await listen("show-overlay", async (event) => {
        await syncLanguageFromSettings();
        // Apply the user's subtitle font size to the CSS var driving the
        // subtitle area. Read fresh on each show so changes take effect
        // without restarting the overlay window.
        try {
          const result = await commands.getAppSettings();
          if (result.status === "ok") {
            const px =
              SUBTITLE_FONT_PX[result.data.subtitle_font_size ?? "medium"] ??
              18;
            document.documentElement.style.setProperty(
              "--subtitle-font-size",
              `${px}px`,
            );
          }
        } catch {
          /* keep CSS fallback */
        }
        const overlayState = event.payload as OverlayState;
        setState(overlayState);
        setIsVisible(true);
        if (overlayState === "recording" || overlayState === "preparing") {
          setSubtitles("");
        }
      });

      const unlistenHide = await listen("hide-overlay", () => {
        setIsVisible(false);
      });

      // Collapse the per-band spectrum into a single amplitude that drives the
      // orb's ring. Smoothed so the ring breathes rather than jitters.
      const unlistenLevel = await listen<number[]>("mic-level", (event) => {
        const bands = event.payload as number[];
        const peak = bands.length
          ? bands.reduce((m, v) => Math.max(m, v), 0)
          : 0;
        smoothedLevel.current = smoothedLevel.current * 0.6 + peak * 0.4;
        setLevel(smoothedLevel.current);
      });

      const unlistenSubtitles = await listen<string>(
        "subtitle-update",
        (event) => setSubtitles(event.payload),
      );

      return () => {
        unlistenShow();
        unlistenHide();
        unlistenLevel();
        unlistenSubtitles();
      };
    };

    setup();
  }, []);

  const isRecording = state === "recording";
  const isPreparing = state === "preparing";
  const isWorking = state === "transcribing" || state === "processing";
  const canCancel = isRecording || isPreparing;

  // Map mic amplitude to a ring scale (1 → 1.9) and glow opacity.
  const ringScale = 1 + Math.min(1, Math.pow(level, 0.7)) * 0.9;
  const ringOpacity = isRecording
    ? Math.max(0.15, Math.min(0.7, level * 1.6))
    : 0;

  const caption = isPreparing
    ? t("overlay.preparing")
    : state === "transcribing"
      ? t("overlay.transcribing")
      : state === "processing"
        ? t("overlay.processing")
        : state === "armed"
          ? t("overlay.wakeArmed")
          : state === "wake-listening"
            ? t("overlay.wakeListening")
            : state === "wake-activated"
              ? t("overlay.wakeActivated")
              : "";

  return (
    <div className="overlay-container">
      <div
        dir={direction}
        className={`orb-overlay state-${state} ${isVisible ? "fade-in" : ""}`}
      >
        <button
          type="button"
          className="orb"
          aria-label={canCancel ? t("overlay.cancel") : caption}
          onClick={() => {
            if (canCancel) commands.cancelOperation();
          }}
        >
          {/* Audio-reactive halo (recording only) */}
          <span
            className="orb-ring"
            aria-hidden="true"
            style={{
              transform: `scale(${ringScale})`,
              opacity: ringOpacity,
            }}
          />
          {/* Indeterminate spinner arc (preparing / transcribing / processing) */}
          {(isPreparing || isWorking) && (
            <span className="orb-spinner" aria-hidden="true" />
          )}
          {/* Core */}
          <span className="orb-core" aria-hidden="true" />
        </button>

        {caption && <div className="orb-caption">{caption}</div>}
      </div>

      {isVisible && subtitles && (
        <div className="subtitle-area fade-in">{subtitles}</div>
      )}
    </div>
  );
};

export default RecordingOverlay;
