import { FileDrop } from "../../components/FileDrop";

export default function UploadPage() {
  return (
    <section className="card">
      <h2>Upload a Dataset</h2>
      <p>
        Drag a file into the drop zone below to upload a dataset to the prediction service. The
        backend will ingest the file and make it available for training.
      </p>
      <FileDrop />
    </section>
  );
}
