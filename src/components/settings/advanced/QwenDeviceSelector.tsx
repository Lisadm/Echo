import React from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../../hooks/useSettings";
import { type QwenDeviceSetting } from "@/bindings";
import { Dropdown } from "../../ui/Dropdown";
import { SettingContainer } from "../../ui/SettingContainer";

interface QwenDeviceSelectorProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

export const QwenDeviceSelector: React.FC<QwenDeviceSelectorProps> = ({
  descriptionMode = "inline",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting } = useSettings();

  const deviceOptions = [
    {
      value: "auto" as QwenDeviceSetting,
      label: t("settings.advanced.qwen.deviceOptions.auto"),
    },
    {
      value: "cuda" as QwenDeviceSetting,
      label: t("settings.advanced.qwen.deviceOptions.cuda"),
    },
    {
      value: "cpu" as QwenDeviceSetting,
      label: t("settings.advanced.qwen.deviceOptions.cpu"),
    },
  ];

  return (
    <SettingContainer
      title={t("settings.advanced.qwen.title")}
      description={t("settings.advanced.qwen.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
    >
      <Dropdown
        options={deviceOptions}
        selectedValue={getSetting("qwen_device") ?? "auto"}
        onSelect={(value) => updateSetting("qwen_device", value as QwenDeviceSetting)}
        disabled={false}
      />
    </SettingContainer>
  );
};
