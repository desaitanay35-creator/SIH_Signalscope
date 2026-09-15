import {
  Clock3,
  FileImage,
  History as HistoryIcon,
  Loader2,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import PageLayout, { PageSection } from "../components/PageLayout";
import AnalysisResult from "../components/AnalysisResult";
import { getAnalysis, listAnalyses } from "../lib/api/analysis";
import { toApiError } from "../lib/api/client";
import type {
  AnalysisListItemResponse,
  AnalysisResponse,
  ApiError,
} from "../lib/api/types";

function verdictLabel(verdict: AnalysisListItemResponse["verdict"]): string {
  return verdict === "likely_ai_generated" ? "Likely AI-generated" : "Likely authentic";
}

export default function History() {
  const [items, setItems] = useState<AnalysisListItemResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const pageSize = 20;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const [openId, setOpenId] = useState<string | null>(null);
  const [openDetail, setOpenDetail] = useState<AnalysisResponse | null>(null);
  const [openLoading, setOpenLoading] = useState(false);

  const load = async (targetPage: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await listAnalyses(targetPage, pageSize);
      setItems(data.items);
      setTotal(data.total);
      setPage(data.page);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const review = async (analysisId: string) => {
    if (openId === analysisId) {
      setOpenId(null);
      setOpenDetail(null);
      return;
    }
    setOpenId(analysisId);
    setOpenDetail(null);
    setOpenLoading(true);
    try {
      const detail = await getAnalysis(analysisId);
      setOpenDetail(detail);
    } catch (err) {
      setError(toApiError(err));
      setOpenId(null);
    } finally {
      setOpenLoading(false);
    }
  };

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <PageLayout
      eyebrow="ANALYSIS HISTORY"
      title="Your review trail."
      copy="Past analyses persisted by the SignalScope backend, most recent first."
    >
      <PageSection title="Recent analyses">
        <div className="history-toolbar">
          <span>
            <HistoryIcon size={15} /> {total} saved assessment
            {total === 1 ? "" : "s"}
          </span>
        </div>

        {loading && (
          <div className="history-empty">
            <Loader2 size={28} className="spin" />
            <h3>Loading history...</h3>
          </div>
        )}

        {!loading && error && (
          <div className="error-message">{error.message}</div>
        )}

        {!loading && !error && items.length === 0 && (
          <div className="history-empty">
            <FileImage size={28} />
            <h3>No analyses yet</h3>
            <p>Run an image through the workspace to see it here.</p>
            <a className="primary-button" href="/analyze">
              Analyze an image
            </a>
          </div>
        )}

        {!loading && !error && items.length > 0 && (
          <>
            <div className="history-list">
              {items.map(item => (
                <div key={item.analysis_id}>
                  <div className="history-row">
                    <div className="history-file">
                      <span className="history-icon">
                        <FileImage size={17} />
                      </span>
                      <div>
                        <strong>{item.filename || "Unnamed image"}</strong>
                        <span>
                          <Clock3 size={12} />{" "}
                          {new Date(item.created_at).toLocaleString()}
                        </span>
                      </div>
                    </div>
                    <div className="history-verdict">
                      <strong>{verdictLabel(item.verdict)}</strong>
                      <span>{Math.round(item.confidence * 100)}% confidence</span>
                    </div>
                    <button
                      className="text-button"
                      onClick={() => review(item.analysis_id)}
                    >
                      {openId === item.analysis_id ? (
                        <>
                          Close <X size={14} />
                        </>
                      ) : (
                        <>Review →</>
                      )}
                    </button>
                  </div>
                  {openId === item.analysis_id && (
                    <div className="history-detail">
                      {openLoading && (
                        <div className="history-empty">
                          <Loader2 size={22} className="spin" />
                        </div>
                      )}
                      {!openLoading && openDetail && (
                        <AnalysisResult
                          result={openDetail}
                          fileName={item.filename}
                          onReset={() => {
                            setOpenId(null);
                            setOpenDetail(null);
                          }}
                        />
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
            {totalPages > 1 && (
              <div className="history-toolbar">
                <button
                  className="secondary-button"
                  disabled={page <= 1}
                  onClick={() => load(page - 1)}
                >
                  Previous
                </button>
                <span>
                  Page {page} of {totalPages}
                </span>
                <button
                  className="secondary-button"
                  disabled={page >= totalPages}
                  onClick={() => load(page + 1)}
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </PageSection>
    </PageLayout>
  );
}
