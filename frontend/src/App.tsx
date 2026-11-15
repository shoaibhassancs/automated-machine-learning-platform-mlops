import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Loading } from "./components/States";
import { PageHeader } from "./components/ui";

const Overview = lazy(() => import("./features/overview/Overview"));
const DatasetList = lazy(() => import("./features/datasets/DatasetList"));
const DatasetVersion = lazy(() => import("./features/datasets/DatasetVersion"));
const JobList = lazy(() => import("./features/training/JobList"));
const NewJob = lazy(() => import("./features/training/NewJob"));
const JobDetail = lazy(() => import("./features/training/JobDetail"));
const Experiments = lazy(() => import("./features/experiments/Experiments"));
const Compare = lazy(() => import("./features/experiments/Compare"));
const ModelList = lazy(() => import("./features/models/ModelList"));
const ModelDetail = lazy(() => import("./features/models/ModelDetail"));
const VersionDetail = lazy(() => import("./features/models/VersionDetail"));
const Deployment = lazy(() => import("./features/deployment/Deployment"));
const TryIt = lazy(() => import("./features/deployment/TryIt"));
const MonitoringOverview = lazy(() => import("./features/monitoring/MonitoringOverview"));
const Reports = lazy(() => import("./features/monitoring/Reports"));
const ReportDetail = lazy(() => import("./features/monitoring/ReportDetail"));
const Alerts = lazy(() => import("./features/monitoring/Alerts"));
const AlertDetail = lazy(() => import("./features/monitoring/AlertDetail"));
const Predictions = lazy(() => import("./features/monitoring/Predictions"));

function NotFound() {
  return (
    <PageHeader
      title="Page not found"
      lead="That address does not exist. Use the menu to continue."
    />
  );
}

export function App() {
  return (
    <Suspense fallback={<Loading label="Loading page" />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="datasets" element={<DatasetList />} />
          <Route path="datasets/:name/v/:number/*" element={<DatasetVersion />} />
          <Route path="training" element={<JobList />} />
          <Route path="training/new" element={<NewJob />} />
          <Route path="training/:id" element={<JobDetail />} />
          <Route path="experiments" element={<Experiments />} />
          <Route path="experiments/compare" element={<Compare />} />
          <Route path="models" element={<ModelList />} />
          <Route path="models/:name" element={<ModelDetail />} />
          <Route path="models/:name/v/:number" element={<VersionDetail />} />
          <Route path="deployment" element={<Deployment />} />
          <Route path="deployment/try" element={<TryIt />} />
          <Route path="monitoring" element={<MonitoringOverview />} />
          <Route path="monitoring/reports" element={<Reports />} />
          <Route path="monitoring/reports/:id" element={<ReportDetail />} />
          <Route path="monitoring/alerts" element={<Alerts />} />
          <Route path="monitoring/alerts/:id" element={<AlertDetail />} />
          <Route path="monitoring/predictions" element={<Predictions />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
