"use client";

import { useEffect, useRef, useState } from "react";

const MAX_CLAIM_FILE_BYTES = 5 * 1024 * 1024;
const MAX_SUPPORTING_FILE_BYTES = 15 * 1024 * 1024;
// Keep local browser requests same-origin so the Next.js dev server can proxy them
// to FastAPI without exposing a second browser-visible port.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

type RecentChat = {
  id: string;
  title: string;
};

type ChartRow = {
  rowId: string;
  rowIndex: number;
  serialNumber?: number;
  patentClaim: string;
  evidenceOfInfringement: string;
  aiEvidence: string;
  confidenceScore?: number;
  confidenceRemark?: string;
  source?: { filename?: string; rowNumber?: number };
  imageRefs?: { assetId: string; filename: string; kind: string }[];
};

type ConfidenceStatus = "idle" | "generating" | "ready" | "error";
type ProcessingStatus = "idle" | "processing" | "ready" | "error";
type AssistantSource = {
  sourceId?: string;
  type?: string;
  label?: string;
  filename?: string;
  title?: string;
  locator?: string | number;
  url?: string;
  snippet?: string;
};

type TablePatch = {
  rowId: string;
  rowIndex: number;
  column: string;
  oldValue: string;
  newValue: string;
  reason?: string;
  sourceRefs?: string[];
};

type AssistantAction = {
  actionId?: string;
  type?: string;
  status?: "pending" | "applying" | "applied" | "rejected";
  patches?: TablePatch[];
  sources?: AssistantSource[];
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: string;
  file_ids?: string[];
  kind?: string;
  action_id?: string;
  action?: AssistantAction;
  sources?: AssistantSource[];
};

type SupportingAttachment = {
  clientKey: string;
  name: string;
  file: File;
  fileId?: string;
};

function fileNameKey(file: { name: string }) {
  return file.name.trim().toLowerCase();
}

function LogoMark() {
  return (
    <svg className="logo-mark" viewBox="0 0 52 52" aria-hidden="true">
      <g className="logo-rays">
        <path d="M26 3v10M26 39v10M3 26h10M39 26h10M9.8 9.8l7.1 7.1M35.1 35.1l7.1 7.1M42.2 9.8l-7.1 7.1M16.9 35.1l-7.1 7.1" />
      </g>
      <g className="logo-sparks">
        <circle cx="26" cy="6" r="2.8" />
        <circle cx="26" cy="46" r="2.8" />
        <circle cx="6" cy="26" r="2.8" />
        <circle cx="46" cy="26" r="2.8" />
        <circle cx="11.8" cy="11.8" r="2.8" />
        <circle cx="40.2" cy="40.2" r="2.8" />
        <circle cx="40.2" cy="11.8" r="2.8" />
        <circle cx="11.8" cy="40.2" r="2.8" />
      </g>
      <circle className="logo-core" cx="26" cy="26" r="5.2" />
    </svg>
  );
}

function PlusIcon() {
  return <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 4v12M4 10h12" /></svg>;
}

function DocumentIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 3.5h8l4 4V20.5H6zM14 3.5v4h4M9 12h6M9 15.5h6" />
    </svg>
  );
}

function PatentDocumentIcon() {
  return (
    <svg className="patent-document-icon" viewBox="0 0 120 120" aria-hidden="true">
      <path className="patent-document-sheet" d="M34 18h38l15 15v64a8 8 0 0 1-8 8H34a8 8 0 0 1-8-8V26a8 8 0 0 1 8-8Z" />
      <path className="patent-document-fold" d="M72 18v17h15" />
      <path className="patent-document-line" d="M42 49h28M42 60h28M42 71h17" />
      <circle className="patent-seal" cx="76" cy="82" r="14" />
      <path className="patent-seal-mark" d="m76 72 2.6 6 6.4.7-4.8 4.3 1.4 6.3-5.6-3.2-5.6 3.2 1.4-6.3-4.8-4.3 6.4-.7Z" />
    </svg>
  );
}

function UploadArrowIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V5M8 9l4-4 4 4M5 16.5v2h14v-2" /></svg>;
}

function PaperclipIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" /></svg>;
}

function SendIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 4 16 8-16 8 3.5-8L4 4Z" /><path d="M7.5 12H20" /></svg>;
}

function ArrowRightIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h13M13 6l6 6-6 6" /></svg>;
}

function DownloadIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11M8 11l4 4 4-4M5 19.5h14" /></svg>;
}

function tableColumnLabel(column: string) {
  if (column === "patentClaim") return "Patent Claim";
  if (column === "evidenceOfInfringement") return "Evidence of Infringement";
  if (column === "aiEvidence") return "AI Evidence";
  return column;
}

function sourceLabel(source: AssistantSource) {
  return source.label || source.filename || source.title || source.sourceId || "Source";
}

export default function Home() {
  const [recentChats, setRecentChats] = useState<RecentChat[]>([]);
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [claimFiles, setClaimFiles] = useState<File[]>([]);
  const [supportingFiles, setSupportingFiles] = useState<SupportingAttachment[]>([]);
  const [claimUploadError, setClaimUploadError] = useState<string | null>(null);
  const [supportingUploadError, setSupportingUploadError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [processedChart, setProcessedChart] = useState<ChartRow[] | null>(null);
  const [confidenceStatus, setConfidenceStatus] = useState<ConfidenceStatus>("idle");
  const [confidenceActiveRow, setConfidenceActiveRow] = useState<number | null>(null);
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus>("idle");
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [fileNamesById, setFileNamesById] = useState<Record<string, string>>({});
  const [editingEvidenceRowId, setEditingEvidenceRowId] = useState<string | null>(null);
  const [editingEvidenceValue, setEditingEvidenceValue] = useState("");
  const [isSavingEvidence, setIsSavingEvidence] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const claimInputRef = useRef<HTMLInputElement>(null);
  const supportingInputRef = useRef<HTMLInputElement>(null);
  const activeRunIdRef = useRef<string | null>(null);
  const workspaceGenerationRef = useRef(0);
  const tableAbortRef = useRef<AbortController | null>(null);
  const confidenceAbortRef = useRef<AbortController | null>(null);
  const assistantAbortRef = useRef<AbortController | null>(null);
  const pendingAttachmentUploadsRef = useRef(0);
  const messageSendLockRef = useRef(false);
  const messageSendTokenRef = useRef(0);
  const [isUploadingAttachments, setIsUploadingAttachments] = useState(false);
  const [actionInFlight, setActionInFlight] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE_URL}/api/chats`)
      .then((response) => (response.ok ? response.json() : []))
      .then((chats: RecentChat[]) => {
        if (!cancelled) setRecentChats(chats);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const invalidateActiveWork = () => {
    workspaceGenerationRef.current += 1;
    tableAbortRef.current?.abort();
    confidenceAbortRef.current?.abort();
    assistantAbortRef.current?.abort();
    tableAbortRef.current = null;
    confidenceAbortRef.current = null;
    assistantAbortRef.current = null;
    activeRunIdRef.current = null;
  };

  const resetComposer = () => {
    invalidateActiveWork();
    messageSendTokenRef.current += 1;
    messageSendLockRef.current = false;
    setClaimFiles([]);
    setSupportingFiles([]);
    setClaimUploadError(null);
    setSupportingUploadError(null);
    setPrompt("");
    setSubmitError(null);
    setIsSendingMessage(false);
    setActiveChat(null);
    setProcessedChart(null);
    setConfidenceStatus("idle");
    setConfidenceActiveRow(null);
    setEditingEvidenceRowId(null);
    setEditingEvidenceValue("");
    setIsExporting(false);
    setExportError(null);
    setProcessingStatus("idle");
    setAssistantOpen(false);
    setMessages([]);
    setFileNamesById({});
  };

  const waitForTable = async (chatId: string, runId: string, generation: number) => {
    const controller = new AbortController();
    tableAbortRef.current = controller;
    try {
      for (let attempt = 0; attempt < 600; attempt += 1) {
        const response = await fetch(
          `${API_BASE_URL}/api/chats/${chatId}/table?run_id=${encodeURIComponent(runId)}`,
          { signal: controller.signal },
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || "The chart table could not be loaded.");
        if (activeRunIdRef.current !== runId || workspaceGenerationRef.current !== generation) return;
        if (payload.processing_status === "completed") {
          setProcessedChart(payload.chart_rows ?? []);
          setProcessingStatus("ready");
          if (payload.chart_rows?.length > 0) void generateConfidence(chatId, runId, generation);
          return;
        }
        if (payload.processing_status === "failed") {
          throw new Error(payload.error || "The chart could not be prepared.");
        }
        await new Promise((resolve) => setTimeout(resolve, 800));
      }
      throw new Error("Chart preparation is taking longer than expected.");
    } catch (error) {
      if (controller.signal.aborted || activeRunIdRef.current !== runId || workspaceGenerationRef.current !== generation) return;
      setProcessingStatus("error");
      setSubmitError(error instanceof Error ? error.message : "The chart could not be prepared.");
    } finally {
      if (tableAbortRef.current === controller) tableAbortRef.current = null;
    }
  };

  const waitForAssistantResponse = async (
    chatId: string,
    runId: string,
    messageId: string,
    generation: number,
  ) => {
    const controller = new AbortController();
    assistantAbortRef.current = controller;
    try {
      for (let attempt = 0; attempt < 600; attempt += 1) {
        const response = await fetch(
          `${API_BASE_URL}/api/chats/${chatId}/messages?run_id=${encodeURIComponent(runId)}`,
          { signal: controller.signal },
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || "The assistant response could not be loaded.");
        if (activeRunIdRef.current !== runId || workspaceGenerationRef.current !== generation) return;
        const nextMessages = (payload as ChatMessage[]) ?? [];
        setMessages(nextMessages);
        const submittedMessage = nextMessages.find((message) => message.id === messageId);
        if (submittedMessage && !["queued", "processing"].includes(submittedMessage.status ?? "")) return;
        await new Promise((resolve) => setTimeout(resolve, 700));
      }
      throw new Error("The assistant is taking longer than expected. Please try again.");
    } catch (error) {
      if (controller.signal.aborted || activeRunIdRef.current !== runId || workspaceGenerationRef.current !== generation) return;
      setSubmitError(error instanceof Error ? error.message : "The assistant response could not be loaded.");
    } finally {
      if (assistantAbortRef.current === controller) assistantAbortRef.current = null;
    }
  };

  const loadWorkspace = async (chatId: string) => {
    invalidateActiveWork();
    const generation = workspaceGenerationRef.current;
    setActiveChat(chatId);
    setSubmitError(null);
    setSupportingUploadError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/chats/${chatId}/workspace`);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "The chart workspace could not be loaded.");
      if (workspaceGenerationRef.current !== generation) return;
      activeRunIdRef.current = payload.run_id;
      const referencedFileIds = new Set<string>(
        (payload.messages ?? []).flatMap((message: ChatMessage) => message.file_ids ?? []),
      );
      setFileNamesById(Object.fromEntries(
        (payload.attachments ?? []).map((attachment: { id: string; filename: string }) => [attachment.id, attachment.filename]),
      ));
      setClaimFiles([]);
      setSupportingFiles((payload.attachments ?? [])
        .filter((attachment: { id: string }) => !referencedFileIds.has(attachment.id))
        .map((attachment: { id: string; filename: string }, index: number) => ({
        clientKey: `persisted-${attachment.id}-${index}`,
        name: attachment.filename,
        file: new File([], attachment.filename),
        fileId: attachment.id,
        })));
      setMessages(payload.messages ?? []);
      setEditingEvidenceRowId(null);
      setEditingEvidenceValue("");
      setProcessedChart(payload.processing_status === "completed" ? (payload.chart_rows ?? []) : null);
      setProcessingStatus(payload.processing_status === "completed" ? "ready" : payload.processing_status === "failed" ? "error" : "processing");
      setConfidenceStatus(payload.confidence_status === "completed" ? "ready" : payload.confidence_status === "generating" ? "generating" : "idle");
      setConfidenceActiveRow(null);
      setAssistantOpen(true);
      const pendingMessage = (payload.messages ?? []).find(
        (message: ChatMessage) => ["queued", "processing"].includes(message.status ?? ""),
      );
      if (payload.processing_status === "processing") {
        void waitForTable(chatId, payload.run_id, generation);
      } else if (payload.processing_status === "failed") {
        setSubmitError(payload.error || "The chart could not be prepared.");
      } else if (payload.confidence_status !== "completed" && payload.chart_rows?.length > 0) {
        void generateConfidence(chatId, payload.run_id, generation);
      }
      if (payload.processing_status === "completed" && pendingMessage) {
        void waitForAssistantResponse(chatId, payload.run_id, pendingMessage.id, generation);
      }
    } catch (error) {
      if (workspaceGenerationRef.current !== generation) return;
      setProcessingStatus("error");
      setSubmitError(error instanceof Error ? error.message : "The chart workspace could not be loaded.");
    }
  };

  const startChartImport = async (files: File[]) => {
    if (files.length === 0 || isSubmitting) return;
    const generation = workspaceGenerationRef.current + 1;
    workspaceGenerationRef.current = generation;
    tableAbortRef.current?.abort();
    confidenceAbortRef.current?.abort();
    setSubmitError(null);
    setIsSubmitting(true);
    const formData = new FormData();
    files.forEach((file) => formData.append("claim_charts", file, file.name));
    try {
      const response = await fetch(`${API_BASE_URL}/api/chats`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "The chart could not be uploaded.");
      if (workspaceGenerationRef.current !== generation) return;
      activeRunIdRef.current = payload.run_id;
      setRecentChats((currentChats) => [
        { id: payload.chat_id, title: payload.title },
        ...currentChats.filter((chat) => chat.id !== payload.chat_id),
      ]);
      setActiveChat(payload.chat_id);
      setClaimFiles([]);
      setProcessedChart(null);
      setEditingEvidenceRowId(null);
      setEditingEvidenceValue("");
      setProcessingStatus("processing");
      setConfidenceStatus("idle");
      void waitForTable(payload.chat_id, payload.run_id, generation);
    } catch (error) {
      if (workspaceGenerationRef.current !== generation) return;
      setProcessingStatus("error");
      setSubmitError(error instanceof Error ? error.message : "The chart could not be uploaded.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClaimUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files ?? []);
    const existingNames = new Set([...claimFiles, ...supportingFiles].map(fileNameKey));
    const acceptedFiles: File[] = [];
    let duplicateCount = 0;
    let oversizedCount = 0;

    selectedFiles.forEach((file) => {
      const name = fileNameKey(file);
      if (file.size >= MAX_CLAIM_FILE_BYTES) {
        oversizedCount += 1;
      } else if (existingNames.has(name)) {
        duplicateCount += 1;
      } else {
        existingNames.add(name);
        acceptedFiles.push(file);
      }
    });

    const nextClaimFiles = [...claimFiles, ...acceptedFiles];
    setClaimFiles(nextClaimFiles);
    setClaimUploadError([
      oversizedCount > 0 ? "Claim-chart files must be smaller than 5 MB." : "",
      duplicateCount > 0 ? "Duplicate filenames were skipped." : "",
    ].filter(Boolean).join(" ") || null);
    event.target.value = "";
  };

  const handleProceed = () => {
    if (claimFiles.length === 0 || activeRunIdRef.current !== null || isSubmitting) return;
    setSubmitError(null);
    setAssistantOpen(true);
    setProcessingStatus("processing");
    void startChartImport(claimFiles);
  };

  const uploadSupportingFiles = async (files: SupportingAttachment[], chatId: string, runId: string) => {
    pendingAttachmentUploadsRef.current += 1;
    setIsUploadingAttachments(true);
    const formData = new FormData();
    formData.append("run_id", runId);
    files.forEach((attachment) => formData.append("supporting_documents", attachment.file, attachment.name));
    try {
      const response = await fetch(`${API_BASE_URL}/api/chats/${chatId}/attachments`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "The supporting document could not be uploaded.");
      setFileNamesById((currentNames) => Object.fromEntries([
        ...Object.entries(currentNames),
        ...(payload.file_ids ?? []).map((fileId: string, index: number) => [
          fileId,
          payload.filenames?.[index] ?? files[index]?.name ?? "Attached document",
        ]),
      ]));
      setSupportingFiles((currentFiles) => currentFiles.map((attachment) => {
        const index = files.findIndex((file) => file.clientKey === attachment.clientKey);
        return index >= 0 ? { ...attachment, fileId: payload.file_ids?.[index] } : attachment;
      }));
    } catch (error) {
      const keys = new Set(files.map((file) => file.clientKey));
      setSupportingFiles((currentFiles) => currentFiles.filter((file) => !keys.has(file.clientKey)));
      setSupportingUploadError(error instanceof Error ? error.message : "The supporting document could not be uploaded.");
    } finally {
      pendingAttachmentUploadsRef.current -= 1;
      if (pendingAttachmentUploadsRef.current === 0) setIsUploadingAttachments(false);
    }
  };

  const handleSupportingUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files ?? []);
    const existingNames = new Set([
      ...claimFiles.map(fileNameKey),
      ...supportingFiles.map(fileNameKey),
    ]);
    const acceptedFiles: SupportingAttachment[] = [];
    let duplicateCount = 0;
    let oversizedCount = 0;

    selectedFiles.forEach((file) => {
      const name = fileNameKey(file);
      if (file.size >= MAX_SUPPORTING_FILE_BYTES) {
        oversizedCount += 1;
      } else if (existingNames.has(name)) {
        duplicateCount += 1;
      } else {
        existingNames.add(name);
        acceptedFiles.push({
          clientKey: `${file.name}-${file.lastModified}-${Math.random().toString(36).slice(2)}`,
          name: file.name,
          file,
        });
      }
    });

    setSupportingFiles((currentFiles) => {
      return [...currentFiles, ...acceptedFiles];
    });
    setSupportingUploadError([
      oversizedCount > 0 ? "Supporting documents must be smaller than 15 MB." : "",
      duplicateCount > 0 ? "Duplicate filenames were skipped." : "",
    ].filter(Boolean).join(" ") || null);
    event.target.value = "";
    if (acceptedFiles.length > 0 && activeChat && activeRunIdRef.current) {
      void uploadSupportingFiles(acceptedFiles, activeChat, activeRunIdRef.current);
    } else if (acceptedFiles.length > 0) {
      setSupportingUploadError("Upload a claim chart before adding supporting documents.");
    }
  };

  const generateConfidence = async (chatId: string, runId: string, generation: number) => {
    const controller = new AbortController();
    confidenceAbortRef.current = controller;
    setConfidenceStatus("generating");
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chats/${chatId}/confidence/stream?run_id=${encodeURIComponent(runId)}`,
        { signal: controller.signal },
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Confidence generation failed.");
      }
      if (!response.body) throw new Error("Confidence generation stream is unavailable.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let eventName = "message";
      let eventData = "";

      const handleEvent = (name: string, data: string) => {
        if (!data) return;
        if (activeRunIdRef.current !== runId || workspaceGenerationRef.current !== generation) return;
        const payload = JSON.parse(data);
        if (name === "row_started") {
          setConfidenceActiveRow(payload.rowIndex);
        } else if (name === "row_completed") {
          setProcessedChart((currentChart) =>
            currentChart?.map((row) =>
              row.rowIndex === payload.rowIndex ? { ...row, ...payload } : row,
            ) ?? currentChart,
          );
        } else if (name === "completed") {
          setProcessedChart(payload.rows ?? []);
          setConfidenceActiveRow(null);
          setConfidenceStatus("ready");
        } else if (name === "error") {
          throw new Error(payload.message || "Confidence generation failed.");
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) eventName = line.slice(7);
            if (line.startsWith("data: ")) eventData += line.slice(6);
          }
          handleEvent(eventName, eventData);
          eventName = "message";
          eventData = "";
        }
        if (done) break;
      }
    } catch (error) {
      if (controller.signal.aborted || activeRunIdRef.current !== runId || workspaceGenerationRef.current !== generation) return;
      setConfidenceActiveRow(null);
      setConfidenceStatus("error");
      setSubmitError(error instanceof Error ? error.message : "Confidence generation failed.");
    } finally {
      if (confidenceAbortRef.current === controller) confidenceAbortRef.current = null;
    }
  };

  const handleSubmit = async () => {
    if (isSubmitting || messageSendLockRef.current || !prompt.trim()) return;
    if (isUploadingAttachments) {
      setSubmitError("Please wait for the attached documents to finish uploading.");
      return;
    }
    if (supportingFiles.some((file) => !file.fileId)) {
      setSubmitError("Please wait for the attached documents to finish uploading.");
      return;
    }
    if (!activeChat || !activeRunIdRef.current) {
      setSubmitError("Upload a claim chart before sending a prompt.");
      return;
    }

    const submittedPrompt = prompt.trim();
    const submittedFileIds = supportingFiles.flatMap((file) => file.fileId ? [file.fileId] : []);
    const sendToken = messageSendTokenRef.current + 1;
    messageSendTokenRef.current = sendToken;
    messageSendLockRef.current = true;
    setIsSendingMessage(true);
    setSubmitError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/chats/${activeChat}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: activeRunIdRef.current,
          content: submittedPrompt,
          file_ids: submittedFileIds,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "The prompt could not be saved.");
      const submittedRunId = activeRunIdRef.current;
      if (!submittedRunId) throw new Error("The chart run is no longer active.");
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: payload.message_id,
          role: "user",
          content: submittedPrompt,
          status: "queued",
          file_ids: submittedFileIds,
        },
      ]);
      setSupportingFiles((currentFiles) => currentFiles.filter(
        (file) => !file.fileId || !submittedFileIds.includes(file.fileId),
      ));
      setPrompt("");
      await waitForAssistantResponse(
        activeChat,
        submittedRunId,
        payload.message_id,
        workspaceGenerationRef.current,
      );
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "The prompt could not be saved.");
    } finally {
      if (messageSendTokenRef.current === sendToken) {
        messageSendLockRef.current = false;
        setIsSendingMessage(false);
      }
    }
  };

  const exportChart = async () => {
    if (isExporting || !activeChat || !activeRunIdRef.current || !processedChart?.length) return;
    const chatId = activeChat;
    const runId = activeRunIdRef.current;
    setIsExporting(true);
    setExportError(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chats/${chatId}/export.pdf?run_id=${encodeURIComponent(runId)}`,
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "The chart could not be exported.");
      }
      const pdfBlob = await response.blob();
      const downloadUrl = URL.createObjectURL(pdfBlob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = "ilumos-claim-chart.pdf";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "The chart could not be exported.");
    } finally {
      setIsExporting(false);
    }
  };

  const decideAssistantAction = async (actionId: string, decision: "approve" | "reject") => {
    if (!actionId || !activeChat || !activeRunIdRef.current || actionInFlight) return;
    const runId = activeRunIdRef.current;
    const generation = workspaceGenerationRef.current;
    setActionInFlight(actionId);
    setSubmitError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/chats/${activeChat}/assistant/actions/${actionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId, decision }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "The table change could not be decided.");
      if (workspaceGenerationRef.current !== generation || activeRunIdRef.current !== runId) return;
      if (payload.status === "applied") {
        confidenceAbortRef.current?.abort();
        setProcessedChart(payload.chart_rows ?? []);
        setProcessingStatus("ready");
        setConfidenceStatus("idle");
        setConfidenceActiveRow(null);
        if (payload.chart_rows?.length > 0) void generateConfidence(activeChat, runId, generation);
      }
      const messagesResponse = await fetch(
        `${API_BASE_URL}/api/chats/${activeChat}/messages?run_id=${encodeURIComponent(runId)}`,
      );
      const nextMessages = await messagesResponse.json().catch(() => []);
      if (messagesResponse.ok && workspaceGenerationRef.current === generation) setMessages(nextMessages);
    } catch (error) {
      if (workspaceGenerationRef.current !== generation) return;
      setSubmitError(error instanceof Error ? error.message : "The table change could not be decided.");
    } finally {
      setActionInFlight(null);
    }
  };

  const startEvidenceEdit = (row: ChartRow) => {
    if (confidenceStatus === "generating" || isSavingEvidence) return;
    setEditingEvidenceRowId(row.rowId || String(row.rowIndex));
    setEditingEvidenceValue(row.evidenceOfInfringement);
    setSubmitError(null);
  };

  const cancelEvidenceEdit = () => {
    setEditingEvidenceRowId(null);
    setEditingEvidenceValue("");
  };

  const saveEvidenceEdit = async (row: ChartRow) => {
    if (!activeChat || !activeRunIdRef.current || isSavingEvidence) return;
    const value = editingEvidenceValue.trim();
    if (!value) {
      setSubmitError("Evidence cannot be empty.");
      return;
    }
    if (value === row.evidenceOfInfringement) {
      cancelEvidenceEdit();
      return;
    }

    const runId = activeRunIdRef.current;
    const generation = workspaceGenerationRef.current;
    setIsSavingEvidence(true);
    setSubmitError(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chats/${activeChat}/rows/${encodeURIComponent(row.rowId || String(row.rowIndex))}/evidence`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_id: runId, old_value: row.evidenceOfInfringement, value }),
        },
      );
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "The evidence could not be saved.");
      if (workspaceGenerationRef.current !== generation || activeRunIdRef.current !== runId) return;

      confidenceAbortRef.current?.abort();
      setProcessedChart(payload.chart_rows ?? []);
      setEditingEvidenceRowId(null);
      setEditingEvidenceValue("");
      setConfidenceStatus("idle");
      setConfidenceActiveRow(null);
      if (payload.chart_rows?.length > 0) void generateConfidence(activeChat, runId, generation);
    } catch (error) {
      if (workspaceGenerationRef.current !== generation || activeRunIdRef.current !== runId) return;
      setSubmitError(error instanceof Error ? error.message : "The evidence could not be saved.");
    } finally {
      setIsSavingEvidence(false);
    }
  };

  const handleComposerKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSubmit();
    }
  };

  const attachedFiles = [
    ...supportingFiles.map((file, index) => ({
      key: file.clientKey || `supporting-${file.name}-${index}`,
      file,
      remove: () => setSupportingFiles((currentFiles) => currentFiles.filter((currentFile) => currentFile.clientKey !== file.clientKey)),
    })),
  ];

  return (
    <div className="workspace-shell min-h-screen">
      <aside className="sidebar" aria-label="iLumos navigation">
        <div className="sidebar-top">
          <div className="brand" aria-label="iLumos by Lumenci">
            <LogoMark />
            <span className="brand-copy">
              <span className="brand-name">iLumos</span>
              <span className="brand-byline">by Lumenci</span>
            </span>
          </div>

          <button className="new-chat-button" type="button" onClick={resetComposer}>
            <PlusIcon />
            <span>New chart</span>
          </button>
        </div>

        {recentChats.length > 0 && (
          <nav className="recent-section" aria-label="Recent charts">
            <div className="section-heading">
              <span>Recent charts</span>
              <span className="heading-rule" aria-hidden="true" />
            </div>
            <div className="chat-list" role="list">
              {recentChats.map((chat) => {
                const isActive = chat.id === activeChat;
                return (
                  <div role="listitem" key={chat.id}>
                    <button
                      className={`chat-row${isActive ? " is-active" : ""}`}
                      type="button"
                      onClick={() => void loadWorkspace(chat.id)}
                      aria-pressed={isActive}
                    >
                      <span className="chat-icon-wrap"><DocumentIcon /></span>
                      <span className="chat-copy"><span className="chat-title">{chat.title}</span></span>
                      <span className="chat-menu" aria-hidden="true"><span className="chat-active-dot" /></span>
                    </button>
                  </div>
                );
              })}
            </div>
          </nav>
        )}

      </aside>

      <main className="main-panel" aria-label="New chart workspace">
        <div className="center-surface">
          <input
            ref={claimInputRef}
            className="file-input-hidden"
            type="file"
            accept=".xlsx,.pdf"
            multiple
            onChange={handleClaimUpload}
            aria-label="Upload claim chart file"
          />
          <input
            ref={supportingInputRef}
            className="file-input-hidden"
            type="file"
            accept=".docx,.pdf"
            multiple
            onChange={handleSupportingUpload}
            aria-label="Upload supporting documents"
          />

          {processingStatus === "processing" ? (
            <section className="chart-result chart-result-loading" aria-labelledby="chart-loading-heading" aria-busy="true">
              <div className="chart-result-header">
                <div>
                  <span className="chart-result-kicker">New claim chart</span>
                  <h1 id="chart-loading-heading">Preparing your chart</h1>
                  <p>The table will appear here while iLumos processes the uploaded document.</p>
                </div>
                <span className="chart-result-badge">Building table</span>
              </div>
              <div className="chart-table-wrap">
                <table className="chart-table" aria-label="Chart loading preview">
                  <colgroup>
                    <col className="chart-serial-column" />
                    <col />
                    <col />
                    <col />
                  </colgroup>
                  <thead>
                    <tr><th className="chart-serial-column">S. No.</th><th>Patent Claim</th><th>Evidence of Infringement</th><th>AI Evidence</th></tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: 8 }, (_, index) => (
                      <tr key={index}>
                        <td className="chart-serial-cell"><span className="table-shimmer table-shimmer-short" /></td>
                        <td><span className="table-shimmer table-shimmer-wide" /></td>
                        <td><span className="table-shimmer" /></td>
                        <td><span className="table-shimmer table-shimmer-short" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : processedChart === null ? (
            <section className="upload-start" aria-labelledby="upload-heading">
              <span className="start-kicker">Start here · Step 1 of 2</span>
              <PatentDocumentIcon />
              <h1 id="upload-heading">Upload your claim chart</h1>
              <p className="upload-subtitle">Begin by uploading one or more claim-chart files. We’ll turn them into an editable workspace.</p>

              <div className="upload-options">
                <article className="upload-card">
                  <span className="upload-card-icon"><DocumentIcon /></span>
                  <span className="upload-card-copy">
                    <span className="upload-card-title">Claim chart <span className="upload-required">— Required</span></span>
                    <span className="upload-card-subtitle">Add your claim-chart file to begin</span>
                    {claimUploadError && <span className="upload-error" role="alert">{claimUploadError}</span>}
                  </span>
                  <button className="upload-file-button" type="button" onClick={() => claimInputRef.current?.click()}>
                    <UploadArrowIcon />
                    <span>{claimFiles.length ? "Upload more" : "Upload file"}</span>
                  </button>
                </article>
                {claimFiles.length > 0 && (
                  <div className="staged-claim-files">
                    <div className="staged-files-heading">
                      <span>{claimFiles.length} claim chart file{claimFiles.length === 1 ? "" : "s"} ready to build</span>
                      <span>You can add more before starting.</span>
                    </div>
                    <div className="staged-files-row">
                      <div className="file-references" aria-label="Selected claim chart files">
                        {claimFiles.slice(0, 3).map((file, index) => (
                          <span className="file-chip" key={`${file.name}-${file.lastModified}-${index}`}>
                            <span className="file-chip-icon"><DocumentIcon /></span>
                            <span className="file-chip-name" title={file.name}>{file.name}</span>
                          </span>
                        ))}
                        {claimFiles.length > 3 && <span className="file-chip-count">+{claimFiles.length - 3} more</span>}
                      </div>
                    </div>
                  </div>
                )}
                <div className="start-action">
                  <button className="start-process-button" type="button" onClick={handleProceed} disabled={claimFiles.length === 0 || isSubmitting}>
                    <span>{isSubmitting ? "Starting chart preparation…" : "Start building chart"}</span>
                    <ArrowRightIcon />
                  </button>
                  <span className="start-action-hint">{claimFiles.length === 0 ? "Upload a claim chart above to continue" : "Your editable table will open next"}</span>
                </div>
              </div>
            </section>
          ) : (
            <section className="chart-result" aria-labelledby="chart-result-heading">
              <div className="chart-result-header">
                <div>
                  <span className="chart-result-kicker">Combined claim chart</span>
                  <h1 id="chart-result-heading">Extracted chart</h1>
                  <p>{processedChart.length} row{processedChart.length === 1 ? "" : "s"} extracted from the uploaded claim chart documents.</p>
                </div>
                <div className="chart-result-header-actions">
                  <button
                    className="export-button"
                    type="button"
                    onClick={() => void exportChart()}
                    disabled={isExporting || !activeChat || !activeRunIdRef.current || processedChart.length === 0}
                    aria-busy={isExporting}
                  >
                    {isExporting ? <span className="export-spinner" aria-hidden="true" /> : <DownloadIcon />}
                    <span>{isExporting ? "Preparing PDF..." : "Export PDF"}</span>
                  </button>
                  <span className="chart-result-badge">
                    {confidenceStatus === "generating" ? `Scoring row ${confidenceActiveRow ?? ""} of ${processedChart.length}` : confidenceStatus === "ready" ? "Confidence ready" : "Chart ready"}
                  </span>
                </div>
              </div>
              {exportError && <p className="chart-export-error" role="alert">{exportError}</p>}
              <div className="chart-table-wrap">
                <table className="chart-table">
                  <colgroup>
                    <col className="chart-serial-column" />
                    <col />
                    <col />
                    <col />
                  </colgroup>
                  <thead>
                    <tr>
                      <th className="chart-serial-column">S. No.</th>
                      <th>Patent Claim</th>
                      <th>Evidence of Infringement</th>
                      <th>AI Evidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {processedChart.map((row) => {
                      const rowId = row.rowId || String(row.rowIndex);
                      const serialNumber = row.serialNumber ?? row.rowIndex;
                      return (
                      <tr key={rowId}>
                        <td className="chart-serial-cell" aria-label={`Serial number ${serialNumber}`}>{serialNumber}</td>
                        <td>{row.patentClaim || "—"}</td>
                        <td>
                          {editingEvidenceRowId === rowId ? (
                            <div className="evidence-editor">
                              <textarea
                                aria-label={`Edit evidence for row ${row.rowIndex}`}
                                rows={4}
                                value={editingEvidenceValue}
                                onChange={(event) => setEditingEvidenceValue(event.target.value)}
                                disabled={isSavingEvidence}
                              />
                              <div className="evidence-editor-actions">
                                <button
                                  className="evidence-save-button"
                                  type="button"
                                  onClick={() => void saveEvidenceEdit(row)}
                                  disabled={isSavingEvidence}
                                >
                                  {isSavingEvidence ? "Saving…" : "Save evidence"}
                                </button>
                                <button
                                  className="evidence-cancel-button"
                                  type="button"
                                  onClick={cancelEvidenceEdit}
                                  disabled={isSavingEvidence}
                                >
                                  Cancel
                                </button>
                              </div>
                            </div>
                          ) : (
                            <div className="evidence-cell-display">
                              <span>{row.evidenceOfInfringement || "—"}</span>
                              <button
                                className="evidence-edit-button"
                                type="button"
                                onClick={() => startEvidenceEdit(row)}
                                disabled={confidenceStatus === "generating" || isSavingEvidence}
                                aria-label={`Edit evidence for row ${row.rowIndex}`}
                              >
                                Edit
                              </button>
                            </div>
                          )}
                          {row.imageRefs && row.imageRefs.length > 0 && (
                            <span className="row-images" aria-label="Source images">
                              {row.imageRefs.map((image) => (
                                <a
                                  className="row-image-link"
                                  href={`${API_BASE_URL}/api/assets/${image.assetId}`}
                                  target="_blank"
                                  rel="noreferrer"
                                  key={image.assetId}
                                >
                                  <img src={`${API_BASE_URL}/api/assets/${image.assetId}`} alt={image.filename} loading="lazy" />
                                  <span>View image</span>
                                </a>
                              ))}
                            </span>
                          )}
                        </td>
                        <td>
                          {typeof row.confidenceScore === "number" ? (
                            <span className="confidence-cell">
                              <strong className="confidence-score">{row.confidenceScore}% confidence</strong>
                              <span className="confidence-remark">{row.confidenceRemark}</span>
                            </span>
                          ) : confidenceStatus === "generating" ? (
                            <span className="confidence-pending"><span className="confidence-spinner" aria-hidden="true" />Waiting for row review…</span>
                          ) : row.aiEvidence || ""}
                        </td>
                      </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}

        </div>

        {assistantOpen && (
          <aside className="assistant-panel" aria-label="iLumos AI assistant">
            <div className="assistant-panel-header">
              <div>
                <span className="assistant-kicker">AI workspace</span>
                <h2>Chart assistant</h2>
              </div>
              <span className={`assistant-status${processingStatus === "processing" ? " is-processing" : ""}`}>
                <span className="assistant-status-dot" />
                {processingStatus === "processing" ? "Preparing chart" : "Ready to help"}
              </span>
            </div>

            <div className="assistant-history" aria-label="Chat history">
              {processingStatus === "processing" && (
                <div className="assistant-message assistant-message-system">
                  <span className="assistant-avatar"><LogoMark /></span>
                  <div>
                    <strong>iLumos</strong>
                    <p>I’m preparing the claim-chart table. You can add supporting files or queue an instruction here while it runs.</p>
                  </div>
                </div>
              )}
              {messages.length === 0 && processingStatus !== "processing" && (
                <div className="assistant-empty-state">
                  <span className="assistant-empty-icon"><PaperclipIcon /></span>
                  <p>Your chart conversation will appear here.</p>
                </div>
              )}
              {messages.map((message) => (
                <div className={`assistant-message assistant-message-${message.role}`} key={message.id}>
                  {message.role === "assistant" && <span className="assistant-avatar"><LogoMark /></span>}
                  <div>
                    <strong>{message.role === "user" ? "You" : "iLumos"}</strong>
                    {message.role === "user" && message.file_ids && message.file_ids.length > 0 && (
                      <div className="message-file-references" aria-label="Message attachments">
                        {message.file_ids.map((fileId) => (
                          <span className="message-file-chip" key={fileId}>
                            <DocumentIcon />
                            <span>{fileNamesById[fileId] ?? "Attached document"}</span>
                          </span>
                        ))}
                      </div>
                    )}
                    {message.role === "user" && message.file_ids && message.file_ids.length > 0 && (
                      <span className="message-file-count" aria-hidden="true">{message.file_ids.length} file{message.file_ids.length === 1 ? "" : "s"} attached</span>
                    )}
                    <p>{message.content}</p>
                    {message.status === "queued" && <span className="message-status">Queued</span>}
                    {message.status === "processing" && <span className="message-status">Reviewing the table…</span>}
                    {message.action?.patches && message.action.patches.length > 0 && (
                      <div className="assistant-proposal" aria-label="Proposed table changes">
                        <div className="assistant-proposal-heading">
                          <span className="assistant-proposal-title">Proposed table changes</span>
                          <span className={`assistant-proposal-state is-${message.action.status ?? "pending"}`}>
                            {message.action.status === "applied" ? "Applied" : message.action.status === "rejected" ? "Rejected" : "Needs approval"}
                          </span>
                        </div>
                        <div className="assistant-change-list">
                          {message.action.patches.map((patch, index) => (
                            <div className="assistant-change" key={`${patch.rowId}-${patch.column}-${index}`}>
                              <div className="assistant-change-meta">Row {patch.rowIndex} · {tableColumnLabel(patch.column)}</div>
                              <div className="assistant-change-old"><span>Current</span><span>{patch.oldValue || "—"}</span></div>
                              <div className="assistant-change-new"><span>Proposed</span><span>{patch.newValue}</span></div>
                              {patch.reason && <div className="assistant-change-reason">{patch.reason}</div>}
                            </div>
                          ))}
                        </div>
                        {(message.action.status ?? "pending") === "pending" && (
                          <div className="assistant-proposal-actions">
                            <button
                              className="assistant-approve"
                              type="button"
                              onClick={() => void decideAssistantAction(message.action_id ?? message.action?.actionId ?? "", "approve")}
                              disabled={actionInFlight !== null}
                            >
                              {actionInFlight === (message.action_id ?? message.action?.actionId) ? "Applying…" : "Approve changes"}
                            </button>
                            <button
                              className="assistant-reject"
                              type="button"
                              onClick={() => void decideAssistantAction(message.action_id ?? message.action?.actionId ?? "", "reject")}
                              disabled={actionInFlight !== null}
                            >
                              Reject
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                    {message.sources && message.sources.length > 0 && (
                      <div className="assistant-sources" aria-label="Sources used">
                        <span className="assistant-sources-label">Sources</span>
                        <div className="assistant-source-list">
                          {message.sources.slice(0, 5).map((source, index) => {
                            const label = sourceLabel(source);
                            const locator = source.locator !== undefined ? ` · ${source.locator}` : "";
                            return source.url ? (
                              <a className="assistant-source" href={source.url} target="_blank" rel="noreferrer" key={`${source.sourceId ?? label}-${index}`}>
                                {label}{locator}
                              </a>
                            ) : (
                              <span className="assistant-source" key={`${source.sourceId ?? label}-${index}`}>
                                {label}{locator}
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <div className="assistant-composer-wrap">
              <div className="chat-composer" role="group" aria-label="Chart prompt">
                {attachedFiles.length > 0 && (
                  <div className="file-references" aria-label="Attached files">
                    {attachedFiles.slice(0, 3).map((attachment) => (
                      <span className="file-chip" key={attachment.key}>
                        <span className="file-chip-icon"><DocumentIcon /></span>
                        <span className="file-chip-name" title={attachment.file.name}>{attachment.file.name}</span>
                        <button className="file-chip-remove" type="button" onClick={attachment.remove} aria-label={`Remove ${attachment.file.name}`}>
                          ×
                        </button>
                      </span>
                    ))}
                    {attachedFiles.length > 3 && <span className="file-chip-count">+{attachedFiles.length - 3} more</span>}
                  </div>
                )}
                <div className="composer-row">
                  <button
                    className="composer-attach"
                    type="button"
                    aria-label="Attach supporting documents"
                    onClick={() => supportingInputRef.current?.click()}
                    disabled={!activeChat || isSubmitting}
                  >
                    <PaperclipIcon />
                  </button>
                  <textarea
                    className="composer-input"
                    aria-label="Ask iLumos to review or refine your chart"
                    placeholder="Ask iLumos to review or refine your chart..."
                    rows={4}
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    onKeyDown={handleComposerKeyDown}
                  />
                  <button
                    className="composer-send"
                    type="button"
                    aria-label={isSendingMessage ? "Sending prompt" : "Send prompt"}
                    onClick={() => void handleSubmit()}
                    disabled={isSubmitting || isSendingMessage || !activeChat || !prompt.trim()}
                    aria-busy={isSubmitting || isSendingMessage}
                  >
                    {isSendingMessage ? <span className="send-spinner" aria-hidden="true" /> : <SendIcon />}
                  </button>
                </div>
              </div>
              <p className={`composer-note${submitError || supportingUploadError ? " composer-error" : ""}`} role={submitError || supportingUploadError ? "alert" : undefined}>
                {submitError || supportingUploadError || "Files are added as references to your prompt."}
              </p>
            </div>
          </aside>
        )}
      </main>
    </div>
  );
}
