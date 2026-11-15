import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState, type DragEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { uploadDataset } from "@/api/client";
import { keys, useDatasets } from "@/api/queries";
import { DataTable } from "@/components/DataTable";
import { Icon } from "@/components/icons";
import { Async, Empty, ErrorPanel } from "@/components/States";
import { useToast } from "@/components/Toast";
import { PageHeader, Panel, Section } from "@/components/ui";
import { bytes, num, when } from "@/lib/format";

const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

export function suggestName(filename: string): string {
  const base = filename
    .replace(/\.[^.]*$/, "")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-");
  return base.replace(/^[^a-z0-9]+/, "").slice(0, 64);
}

function UploadForm() {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const client = useQueryClient();
  const navigate = useNavigate();
  const toast = useToast();

  const nameError =
    name && !NAME_RE.test(name)
      ? "Use lower-case letters, digits, '-' or '_' (start with a letter or digit, at most 64 characters)."
      : null;

  function pick(f: File | undefined) {
    if (!f) return;
    setFile(f);
    setError(null);
    if (!name) setName(suggestName(f.name));
  }
  function drop(e: DragEvent) {
    e.preventDefault();
    setOver(false);
    pick(e.dataTransfer.files[0]);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !name || nameError) return;
    setBusy(true);
    setError(null);
    try {
      const result = await uploadDataset(file, name);
      await client.invalidateQueries({ queryKey: keys.datasets });
      toast(
        result.created
          ? `Uploaded ${result.version.ref}: ${num(result.version.rows)} rows.`
          : `That exact file was already stored as ${result.version.ref}.`,
      );
      navigate(`/datasets/${result.version.dataset}/v/${result.version.number}`);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel>
      <form onSubmit={submit} className="stack" noValidate>
        <h2>Upload a CSV</h2>
        <div
          className={`dropzone${over ? " over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={drop}
        >
          <Icon.Upload />
          <p>
            <strong>{file ? file.name : "Drop a CSV file here"}</strong>
          </p>
          <p className="muted">{file ? bytes(file.size) : "or choose one from your computer"}</p>
          <div style={{ marginTop: 10 }}>
            <label className="btn" htmlFor="csv-file">
              {file ? "Choose a different file" : "Choose a file"}
            </label>
            <input
              id="csv-file"
              ref={input}
              className="sr-only"
              type="file"
              accept=".csv,text/csv"
              onChange={(e) => pick(e.target.files?.[0])}
            />
          </div>
        </div>
        <div className="field" style={{ maxWidth: 420 }}>
          <label htmlFor="dataset-name">Dataset name</label>
          <input
            id="dataset-name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            aria-invalid={nameError ? true : undefined}
            aria-describedby="dataset-name-hint"
            autoComplete="off"
          />
          <span id="dataset-name-hint" className={nameError ? "error" : "hint"}>
            {nameError ??
              "Uploading the same name again creates a new version; old versions never change."}
          </span>
        </div>
        {error !== null && <ErrorPanel error={error} title="The upload was rejected" />}
        <div>
          <button
            type="submit"
            className="btn primary"
            disabled={!file || !name || !!nameError || busy}
          >
            {busy ? "Uploading…" : "Upload and inspect"}
          </button>
        </div>
      </form>
    </Panel>
  );
}

export default function DatasetList() {
  const query = useDatasets();
  return (
    <div className="stack-lg">
      <PageHeader
        title="Datasets"
        lead="Every upload becomes an immutable version with a checksum. You can preview it, see a column-by-column profile, read the validation findings, and set up the prediction task."
      />
      <UploadForm />
      <Section title="Your datasets">
        <Async query={query} label="Loading datasets">
          {(data) =>
            data.datasets.length === 0 ? (
              <Panel>
                <Empty title="No datasets yet">Upload a CSV above to begin the lifecycle.</Empty>
              </Panel>
            ) : (
              <Panel flush>
                <DataTable
                  caption="Datasets and their versions"
                  hideCaption
                  rows={data.datasets.flatMap((d) => d.versions)}
                  rowKey={(v) => v.ref}
                  columns={[
                    {
                      key: "ref",
                      header: "Dataset version",
                      cell: (v) => (
                        <Link className="row-link" to={`/datasets/${v.dataset}/v/${v.number}`}>
                          {v.dataset} <span className="muted">v{v.number}</span>
                        </Link>
                      ),
                    },
                    { key: "rows", header: "Rows", numeric: true, cell: (v) => num(v.rows) },
                    {
                      key: "cols",
                      header: "Columns",
                      numeric: true,
                      cell: (v) => num(v.columns.length),
                    },
                    {
                      key: "size",
                      header: "Size",
                      numeric: true,
                      cell: (v) => bytes(v.size_bytes),
                    },
                    { key: "file", header: "Source file", cell: (v) => v.source_filename },
                    { key: "at", header: "Uploaded", cell: (v) => when(v.created_at) },
                  ]}
                />
              </Panel>
            )
          }
        </Async>
      </Section>
    </div>
  );
}
