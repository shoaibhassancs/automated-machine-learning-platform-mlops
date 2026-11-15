import type { ReactNode, SVGProps } from "react";

function Svg({ children, ...rest }: SVGProps<SVGSVGElement> & { children: ReactNode }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const Icon = {
  Logo: () => (
    <Svg width="18" height="18" strokeWidth="2.6">
      <path d="M4 17l5-6 4 3 7-9" />
    </Svg>
  ),
  Check: () => (
    <Svg width="14" height="14" strokeWidth="3">
      <path d="M5 13l4 4L19 7" />
    </Svg>
  ),
  Alert: () => (
    <Svg width="14" height="14">
      <path d="M12 3l10 18H2L12 3zM12 10v5M12 18v.01" />
    </Svg>
  ),
  Info: () => (
    <Svg>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v6M12 7.5v.01" />
    </Svg>
  ),
  Warn: () => (
    <Svg>
      <path d="M12 3l10 18H2L12 3zM12 10v5M12 18v.01" />
    </Svg>
  ),
  Cross: () => (
    <Svg width="14" height="14" strokeWidth="3">
      <path d="M6 6l12 12M18 6L6 18" />
    </Svg>
  ),
  Menu: () => (
    <Svg>
      <path d="M4 7h16M4 12h16M4 17h16" />
    </Svg>
  ),
  Upload: () => (
    <Svg width="28" height="28">
      <path d="M12 16V4M7 9l5-5 5 5M4 20h16" />
    </Svg>
  ),
  Clock: () => (
    <Svg width="14" height="14">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </Svg>
  ),
  Dot: () => (
    <Svg width="10" height="10" fill="currentColor" stroke="none">
      <circle cx="12" cy="12" r="8" />
    </Svg>
  ),
  Help: () => (
    <Svg width="14" height="14">
      <circle cx="12" cy="12" r="9" />
      <path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7M12 17v.01" />
    </Svg>
  ),
};
