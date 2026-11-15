export type Status = "development" | "staging" | "production" | "archived";

/** The registry's rules (docs/registry.md): what a person may do with a version in each status. */
export function allowedActions(status: Status): {
  promote: ("staging" | "production")[];
  archive: boolean;
} {
  switch (status) {
    case "development":
      return { promote: ["staging"], archive: true };
    case "staging":
      return { promote: ["production"], archive: true };
    case "archived":
      return { promote: ["staging"], archive: false };
    default:
      return { promote: [], archive: false };
  }
}

export const STATUS_HELP: Record<Status, string> = {
  development: "Just registered. Not yet checked for use.",
  staging: "Being checked before it is used for real.",
  production: "The version treated as live. Only one version is in production.",
  archived: "Retired. Kept for the record; can be promoted back to staging.",
};
