import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../../hooks/useSettings";
import { commands, type InputMode } from "@/bindings";
import { Dropdown } from "../../ui/Dropdown";
import { Input } from "../../ui/Input";
import { SettingContainer } from "../../ui/SettingContainer";

interface WakeListeningProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

const MODE_OPTIONS: { value: InputMode; key: string }[] = [
  { value: "normal_dictation", key: "normal_dictation" },
  { value: "wake_always_listening", key: "wake_always_listening" },
  { value: "wake_hotkey_toggle", key: "wake_hotkey_toggle" },
];

export const WakeListeningSettings: React.FC<WakeListeningProps> = ({
  descriptionMode = "inline",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const [phraseDraft, setPhraseDraft] = useState<string | null>(null);
  const [alternativesDraft, setAlternativesDraft] = useState<string | null>(
    null,
  );

  const modeValue = (getSetting("input_mode") ?? "normal_dictation") as InputMode;
  const wakePhrase = (getSetting("wake_phrase") ?? "").toString();
  const timeout = Number(getSetting("wake_activation_timeout_secs") ?? 7);
  const enabled = modeValue !== "normal_dictation";

  // Text fields are stored server-side on commit (blur) so typing isn't
  // fought by async setting round-trips; drafts reset when settings change.
  const phraseValue = phraseDraft ?? wakePhrase;

  const commitPhrase = (draft: string) => {
    setPhraseDraft(null);
    const v = draft.trim();
    if (v && v !== wakePhrase) {
      commands
        .changeWakePhraseSetting(v)
        .then(() => updateSetting("wake_phrase", v))
        .catch((err) => console.error("Failed to update wake phrase:", err));
    }
  };

  // Alternatives are stored as a string[]; edit as a comma-separated draft
  // committed on blur so typing commas isn't fought by the updater.
  const alternativesValue =
    alternativesDraft ??
    (getSetting("wake_alternative_phrases") ?? []).join(", ");

  const commitAlternatives = (draft: string) => {
    setAlternativesDraft(null);
    const list = draft
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    updateSetting("wake_alternative_phrases", list);
  };

  return (
    <>
      <SettingContainer
        title={t("settings.advanced.wakeListening.title")}
        description={t("settings.advanced.wakeListening.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      >
        <Dropdown
          options={MODE_OPTIONS.map((o) => ({
            value: o.value,
            label: t(
              `settings.advanced.wakeListening.modes.${o.key}`,
            ),
          }))}
          selectedValue={modeValue}
          onSelect={(value) =>
            commands
              .changeInputModeSetting(value as InputMode)
              .then(() => updateSetting("input_mode", value as InputMode))
              .catch((e) =>
                console.error("Failed to update input mode:", e),
              )
          }
          disabled={isUpdating("input_mode")}
        />
      </SettingContainer>

      {enabled && (
        <>
          <SettingContainer
            title={t("settings.advanced.wakeListening.wakePhrase")}
            description={t("settings.advanced.wakeListening.wakePhraseDescription")}
            descriptionMode={descriptionMode}
            grouped={grouped}
          >
            <Input
              type="text"
              value={phraseValue}
              onChange={(e) => setPhraseDraft(e.target.value)}
              onBlur={(e) => commitPhrase(e.target.value)}
              placeholder={t("settings.advanced.wakeListening.wakePhrase")}
            />
          </SettingContainer>

          <SettingContainer
            title={t("settings.advanced.wakeListening.alternatives")}
            description={t("settings.advanced.wakeListening.alternativesDescription")}
            descriptionMode={descriptionMode}
            grouped={grouped}
          >
            <Input
              type="text"
              value={alternativesValue}
              onChange={(e) => setAlternativesDraft(e.target.value)}
              onBlur={(e) => commitAlternatives(e.target.value)}
              placeholder="эхо, эко"
            />
          </SettingContainer>

          <SettingContainer
            title={t("settings.advanced.wakeListening.timeout")}
            description={t("settings.advanced.wakeListening.timeoutDescription")}
            descriptionMode={descriptionMode}
            grouped={grouped}
          >
            <Input
              type="number"
              min={1}
              max={60}
              value={timeout}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v) && v >= 1) {
                  commands
                    .changeWakeActivationTimeoutSetting(v)
                    .then(() => updateSetting("wake_activation_timeout_secs", v))
                    .catch((err) =>
                      console.error("Failed to update wake timeout:", err),
                    );
                }
              }}
            />
          </SettingContainer>
        </>
      )}
    </>
  );
};
