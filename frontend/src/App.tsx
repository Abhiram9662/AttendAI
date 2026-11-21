import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import axios from "axios";
import {
  Camera, CameraOff, ScanFace, ShieldCheck, Users, CalendarDays,
  RefreshCw, CheckCircle2, XCircle, Clock3, Activity, Upload, Trash2
} from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

type Student = {
  id: number;
  student_id: string;
  full_name: string;
  email: string;
  department?: string;
  section?: string;
};

type AttendanceSession = {
  id: number;
  title: string;
  starts_at: string;
  ends_at: string;
  status: string;
};

type Detection = {
  face_count: number;
  faces: { x: number; y: number; width: number; height: number }[];
  notice?: string;
};

type RecognitionFace = {
  face_index: number;
  student_id: string | null;
  status: string;
  similarity: number;
  svm_prediction?: string | null;
  nearest_centroid?: string | null;
  svm_centroid_agreement?: boolean;
  detection_probability?: number;
};

type AttendanceResult = {
  student_id: string;
  status: string;
  attendance_id?: number;
};

type RecognitionResponse = {
  face_count: number;
  recognized_count: number;
  unknown_count: number;
  threshold: number;
  faces: RecognitionFace[];
  attendance: AttendanceResult[];
  notice?: string;
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const enrollVideoRef = useRef<HTMLVideoElement>(null);
  const enrollCanvasRef = useRef<HTMLCanvasElement>(null);
  const enrollStreamRef = useRef<MediaStream | null>(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("Camera is off");
  const [detection, setDetection] = useState<Detection | null>(null);
  const [busy, setBusy] = useState(false);
  const [recognizing, setRecognizing] = useState(false);
  const [students, setStudents] = useState<Student[]>([]);
  const [sessions, setSessions] = useState<AttendanceSession[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [pageError, setPageError] = useState("");
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [lastUpdated, setLastUpdated] = useState("");
  const [studentForm, setStudentForm] = useState({
    student_id: "",
    full_name: "",
    email: "",
    department: "Computer Science",
    section: "CSE"
  });
  const [addingStudent, setAddingStudent] = useState(false);
  const [studentMessage, setStudentMessage] = useState("");
  const [enrollCameraOn, setEnrollCameraOn] = useState(false);
  const [enrollmentPhotos, setEnrollmentPhotos] = useState<Blob[]>([]);
  const [enrollmentCameraError, setEnrollmentCameraError] = useState("");

  const loadDashboard = useCallback(async () => {
    setPageError("");
    try {
      const [usersResponse, sessionsResponse, healthResponse] = await Promise.all([
        axios.get<Student[]>(`${API}/api/vision/students`),
        axios.get<AttendanceSession[]>(`${API}/api/sessions`),
        axios.get(`${API}/api/health`)
      ]);
      setStudents(Array.isArray(usersResponse.data) ? usersResponse.data : []);
      const sessionData = Array.isArray(sessionsResponse.data) ? sessionsResponse.data : [];
      setSessions(sessionData);
      setSessionId(current => {
        if (current && sessionData.some(s => String(s.id) === current)) return current;
        const open = sessionData.find(s => s.status?.toLowerCase() === "open");
        return open ? String(open.id) : (sessionData[0] ? String(sessionData[0].id) : "");
      });
      setApiStatus(healthResponse.data?.status === "ok" ? "online" : "offline");
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      console.error(error);
      setApiStatus("offline");
      setPageError("Could not load dashboard data. Make sure FastAPI is running at " + API);
    }
  }, []);

  useEffect(() => { void loadDashboard(); }, [loadDashboard]);

  async function deleteStudent(student: Student) {
    const confirmed = window.confirm(
      `PERMANENTLY DELETE ${student.full_name} (${student.student_id})? This removes the student from the backend database and deletes their attendance records. This action cannot be undone.`
    );
    if (!confirmed) return;
    setStudentMessage("");
    setPageError("");
    try {
      const response = await axios.delete(`${API}/api/vision/students/${encodeURIComponent(student.student_id)}`);
      setStudentMessage(response.data?.message || `Student ${student.student_id} permanently deleted.`);
      await loadDashboard();
    } catch (error) {
      console.error(error);
      setPageError(axios.isAxiosError(error)
        ? (error.response?.data?.detail || "Could not delete this student from the backend.")
        : "Could not delete this student from the backend.");
    }
  }

  async function startEnrollmentCamera() {
    setEnrollmentCameraError("");
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        setEnrollmentCameraError("This browser does not support webcam access.");
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      enrollStreamRef.current = stream;
      if (enrollVideoRef.current) {
        enrollVideoRef.current.srcObject = stream;
        await enrollVideoRef.current.play();
      }
      setEnrollCameraOn(true);
    } catch (error) {
      console.error(error);
      setEnrollmentCameraError("Camera unavailable. Allow camera access and try again.");
    }
  }

  function stopEnrollmentCamera() {
    enrollStreamRef.current?.getTracks().forEach(track => track.stop());
    enrollStreamRef.current = null;
    if (enrollVideoRef.current) enrollVideoRef.current.srcObject = null;
    setEnrollCameraOn(false);
  }

  async function captureEnrollmentPhoto() {
    const video = enrollVideoRef.current;
    const canvas = enrollCanvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      setEnrollmentCameraError("Wait until the webcam preview is ready.");
      return;
    }
    if (enrollmentPhotos.length >= 30) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) {
      setEnrollmentCameraError("Could not capture this frame. Try again.");
      return;
    }
    setEnrollmentPhotos(current => current.length < 30 ? [...current, blob] : current);
    setEnrollmentCameraError("");
  }

  async function addStudent(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (enrollmentPhotos.length !== 30) {
      setStudentMessage("Capture all 30 face photos before registering this student.");
      return;
    }
    setAddingStudent(true);
    setStudentMessage("Uploading 30 photos…");
    setPageError("");
    try {
      const form = new FormData();
      form.append("student_id", studentForm.student_id.trim());
      form.append("full_name", studentForm.full_name.trim());
      form.append("email", studentForm.email.trim());
      form.append("department", studentForm.department.trim());
      form.append("section", studentForm.section.trim());
      enrollmentPhotos.forEach((photo, index) => {
        form.append("images", photo, `enrollment_${String(index + 1).padStart(2, "0")}.jpg`);
      });
      const response = await axios.post(`${API}/api/vision/enroll`, form);
      setStudentMessage(response.data?.message || "Student and face enrollment completed successfully.");
      setStudentForm({ student_id: "", full_name: "", email: "", department: "Computer Science", section: "CSE" });
      setEnrollmentPhotos([]);
      stopEnrollmentCamera();
      await loadDashboard();
    } catch (error) {
      console.error(error);
      if (axios.isAxiosError(error)) {
        const detail = error.response?.data?.detail;
        setStudentMessage(typeof detail === "string" ? detail : "Enrollment failed. Check the student details and backend terminal.");
      } else {
        setStudentMessage("Enrollment failed. Please try again.");
      }
    } finally {
      setAddingStudent(false);
    }
  }

  async function startCamera() {
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraStatus("This browser does not support camera access.");
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
      setCameraStatus("Camera active — position one face in the frame");
      setResult(null);
    } catch (error) {
      console.error(error);
      setCameraStatus("Camera unavailable or permission denied. Allow camera access in your browser.");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setDetection(null);
    setCameraStatus("Camera is off");
  }

  async function captureImage(): Promise<Blob | null> {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
  }

  const detectFace = useCallback(async () => {
    if (!cameraOn || busy || recognizing) return;
    setBusy(true);
    try {
      const blob = await captureImage();
      if (!blob) return;
      const form = new FormData();
      form.append("image", blob, "camera.jpg");
      const response = await axios.post<Detection>(`${API}/api/vision/detect`, form);
      setDetection(response.data);
      setCameraStatus(response.data.face_count
        ? `${response.data.face_count} face(s) detected`
        : "No face detected — adjust your position");
    } catch (error) {
      console.error(error);
      setCameraStatus("Detection API error. Check the backend.");
    } finally {
      setBusy(false);
    }
  }, [cameraOn, busy, recognizing]);

  useEffect(() => {
    if (!cameraOn) return;
    const timer = window.setInterval(() => { void detectFace(); }, 1400);
    return () => window.clearInterval(timer);
  }, [cameraOn, detectFace]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    enrollStreamRef.current?.getTracks().forEach(track => track.stop());
  }, []);

  async function recognizeAndMark() {
    if (!cameraOn) {
      setCameraStatus("Start the camera first.");
      return;
    }
    if (!sessionId) {
      setPageError("Create an attendance session first, then select it here.");
      return;
    }
    setRecognizing(true);
    setPageError("");
    setResult(null);
    try {
      const blob = await captureImage();
      if (!blob) throw new Error("Camera frame is not ready. Wait a moment and try again.");
      const form = new FormData();
      form.append("image", blob, "attendance-camera.jpg");
      form.append("session_id", sessionId);
      const response = await axios.post<RecognitionResponse>(
        `${API}/api/vision/recognize`,
        form
      );
      setResult(response.data);
      const marked = response.data.attendance?.filter(a => a.status === "recorded").length ?? 0;
      const already = response.data.attendance?.filter(a => a.status === "already_marked").length ?? 0;
      if (marked) setCameraStatus(`Attendance recorded for ${marked} student(s).`);
      else if (already) setCameraStatus("Attendance was already recorded for this session.");
      else setCameraStatus("No registered student was accepted. Check the result below.");
      await loadDashboard();
    } catch (error) {
      console.error(error);
      if (axios.isAxiosError(error)) {
        setPageError(error.response?.data?.detail || "Recognition request failed. Check the selected session and backend terminal.");
      } else {
        setPageError(error instanceof Error ? error.message : "Recognition failed.");
      }
    } finally {
      setRecognizing(false);
    }
  }

  async function recognizeUploadedFile(file: File) {
    if (!sessionId) {
      setPageError("Select an attendance session first.");
      return;
    }
    setRecognizing(true);
    setPageError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("session_id", sessionId);
      const response = await axios.post<RecognitionResponse>(`${API}/api/vision/recognize`, form);
      setResult(response.data);
      await loadDashboard();
    } catch (error) {
      console.error(error);
      setPageError(axios.isAxiosError(error)
        ? (error.response?.data?.detail || "Image recognition failed.")
        : "Image recognition failed.");
    } finally {
      setRecognizing(false);
    }
  }

  const recordedCount = result?.attendance?.filter(a => a.status === "recorded").length ?? 0;
  const duplicateCount = result?.attendance?.filter(a => a.status === "already_marked").length ?? 0;

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brandIcon"><ScanFace /></div>
          <div><strong>AttendAI</strong><small>SMART ATTENDANCE PLATFORM</small></div>
        </div>
        <div className="topActions">
          <span className={`apiPill ${apiStatus}`}><span className="dot" /> API {apiStatus}</span>
          <button className="secondary refreshButton" onClick={() => void loadDashboard()}><RefreshCw size={15}/> Refresh</button>
        </div>
      </header>

      <section className="intro">
        <div className="eyebrow">CSE471 GROUP PROJECT</div>
        <h1>AI-Powered Attendance<br/><span>Management System</span></h1>
        <p>Face recognition, attendance recording and student management — connected to your local FastAPI backend.</p>
      </section>

      {pageError && <div className="errorBanner"><XCircle size={18}/><span>{pageError}</span></div>}

      <section className="statsGrid">
        <div className="statCard"><div className="statIcon"><Users/></div><span>Registered students</span><strong>{students.length}</strong></div>
        <div className="statCard"><div className="statIcon"><CalendarDays/></div><span>Attendance sessions</span><strong>{sessions.length}</strong></div>
        <div className="statCard"><div className="statIcon"><Activity/></div><span>API connection</span><strong className="statText">{apiStatus === "online" ? "Connected" : apiStatus === "checking" ? "Checking…" : "Offline"}</strong></div>
        <div className="statCard"><div className="statIcon"><Clock3/></div><span>Last refreshed</span><strong className="statText">{lastUpdated || "—"}</strong></div>
      </section>

      <section className="workspace">
        <div className="cameraCard">
          <div className="cardHead">
            <div><strong>Face recognition</strong><small>Camera preview and attendance capture</small></div>
            <span className={cameraOn ? "live" : "offline"}>{cameraOn ? "● CAMERA LIVE" : "● CAMERA OFF"}</span>
          </div>

          <label className="fieldLabel" htmlFor="session">Attendance session</label>
          <select id="session" value={sessionId} onChange={e => setSessionId(e.target.value)}>
            <option value="">Select a session</option>
            {sessions.map(session => <option key={session.id} value={session.id}>
              #{session.id} — {session.title} ({session.status})
            </option>)}
          </select>

          <div className="videoArea">
            <video ref={videoRef} playsInline muted className={cameraOn ? "visible" : "hidden"} />
            {!cameraOn && <div className="cameraPlaceholder"><Camera size={42}/><strong>Camera preview</strong><span>Start the camera to begin</span></div>}
          </div>
          <div className="controls">
            <button className="primary" onClick={() => void startCamera()} disabled={cameraOn}><Camera size={17}/> Start camera</button>
            <button className="secondary" onClick={stopCamera} disabled={!cameraOn}><CameraOff size={17}/> Stop camera</button>
            <button className="recognizeButton" onClick={() => void recognizeAndMark()} disabled={!cameraOn || recognizing || !sessionId}>
              <ScanFace size={17}/> {recognizing ? "Recognizing…" : "Recognize & mark"}
            </button>
          </div>
          <div className="status"><span className={cameraOn ? "dot green" : "dot"}></span>{cameraStatus}</div>
          <div className="uploadRow">
            <label className="uploadButton"><Upload size={16}/> Test with an image
              <input type="file" accept="image/jpeg,image/png,image/webp" disabled={recognizing || !sessionId}
                onChange={e => { const file = e.target.files?.[0]; if (file) void recognizeUploadedFile(file); e.currentTarget.value = ""; }}/>
            </label>
            <small>Image upload is useful for testing; camera capture is available above.</small>
          </div>
        </div>

        <aside className="infoCard">
          <div className="infoIcon"><ShieldCheck/></div>
          <h2>Recognition status</h2>
          <p className="muted">Results returned by your trained FaceNet + classifier pipeline</p>
          <div className="metric"><span>Faces detected</span><strong>{result?.face_count ?? detection?.face_count ?? "—"}</strong></div>
          <div className="metric"><span>Recognized faces</span><strong>{result?.recognized_count ?? "—"}</strong></div>
          <div className="metric"><span>Unknown faces</span><strong>{result?.unknown_count ?? "—"}</strong></div>
          <div className="metric"><span>Attendance recorded</span><strong>{result ? recordedCount : "—"}</strong></div>
          <div className="metric"><span>Already marked</span><strong>{result ? duplicateCount : "—"}</strong></div>
          <div className="notice"><strong>Prototype safety note</strong><p>Uploaded images and camera frames are processed by your local API. Liveness verification is not implemented, so a photograph alone cannot prove that a live person is present.</p></div>
        </aside>
      </section>

      {result && <section className="resultsCard">
        <div className="sectionHeading"><div><strong>Latest recognition result</strong><small>Session #{sessionId}</small></div><span className="resultBadge">{result.recognized_count} recognized · {result.unknown_count} unknown</span></div>
        {result.faces.length === 0 ? <p className="muted">No faces were detected in this image.</p> :
          <div className="tableWrap"><table><thead><tr><th>Face</th><th>Student ID</th><th>Status</th><th>Similarity</th><th>Attendance</th></tr></thead>
            <tbody>{result.faces.map(face => {
              const attendance = result.attendance?.find(a => a.student_id === face.student_id);
              return <tr key={face.face_index}>
                <td>#{face.face_index}</td>
                <td>{face.student_id || "—"}</td>
                <td><span className={face.status === "recognized" ? "tag success" : "tag warning"}>{face.status}</span></td>
                <td>{(face.similarity * 100).toFixed(2)}%</td>
                <td>{attendance?.status || (face.status === "unknown" ? "Not recorded" : "—")}</td>
              </tr>;
            })}</tbody></table></div>}
        {result.notice && <p className="resultNotice">{result.notice}</p>}
      </section>}

      <section className="studentsCard enrollmentCard">
        <div className="sectionHeading">
          <div><strong>Register student & enroll face</strong><small>Student details plus 30 mandatory webcam photos</small></div>
          <span className="countBadge">30 photos required</span>
        </div>
        <form className="studentForm" onSubmit={addStudent}>
          <div className="formGrid">
            <label>Student ID
              <input required maxLength={50} value={studentForm.student_id}
                onChange={e => setStudentForm({ ...studentForm, student_id: e.target.value })}
                placeholder="e.g. 12407909" />
            </label>
            <label>Full name
              <input required minLength={2} maxLength={150} value={studentForm.full_name}
                onChange={e => setStudentForm({ ...studentForm, full_name: e.target.value })}
                placeholder="Student full name" />
            </label>
            <label>Email address
              <input required type="email" maxLength={255} value={studentForm.email}
                onChange={e => setStudentForm({ ...studentForm, email: e.target.value })}
                placeholder="student@example.com" />
            </label>
            <label>Department
              <input value={studentForm.department}
                onChange={e => setStudentForm({ ...studentForm, department: e.target.value })}
                placeholder="Computer Science" />
            </label>
            <label>Section
              <input value={studentForm.section}
                onChange={e => setStudentForm({ ...studentForm, section: e.target.value })}
                placeholder="CSE" />
            </label>
          </div>

          <div className="enrollmentCapture">
            <div className="enrollmentCaptureHead">
              <div><strong>Webcam face enrollment</strong><small>Upload exactly 30 student images (face checks disabled)</small></div>
              <span className="photoCounter">{enrollmentPhotos.length}/30</span>
            </div>
            <div className="enrollmentProgress"><span style={{ width: `${(enrollmentPhotos.length / 30) * 100}%` }} /></div>
            <div className="enrollmentVideoArea">
              <video ref={enrollVideoRef} playsInline muted className={enrollCameraOn ? "visible" : "hidden"} />
              {!enrollCameraOn && <div className="cameraPlaceholder"><Camera size={34}/><strong>Enrollment camera is off</strong><span>Start the webcam to capture face samples</span></div>}
            </div>
            <canvas ref={enrollCanvasRef} className="hiddenCanvas" />
            <div className="controls">
              <button className="primary" type="button" onClick={() => void startEnrollmentCamera()} disabled={enrollCameraOn || addingStudent}>
                <Camera size={16}/> Start enrollment camera
              </button>
              <button className="secondary" type="button" onClick={captureEnrollmentPhoto} disabled={!enrollCameraOn || enrollmentPhotos.length >= 30 || addingStudent}>
                <ScanFace size={16}/> Capture photo ({enrollmentPhotos.length}/30)
              </button>
              <button className="secondary" type="button" onClick={() => setEnrollmentPhotos([])} disabled={enrollmentPhotos.length === 0 || addingStudent}>
                Retake all
              </button>
              <button className="secondary" type="button" onClick={stopEnrollmentCamera} disabled={!enrollCameraOn || addingStudent}>
                <CameraOff size={16}/> Stop camera
              </button>
            </div>
            <p className="captureGuidance">Keep only the enrolling student in frame. Capture 30 samples with small natural changes in head angle and expression. Keep lighting clear and avoid masks or sunglasses.</p>
            {enrollmentCameraError && <p className="formError" role="alert">{enrollmentCameraError}</p>}
          </div>

          <button className="primary enrollmentSubmit" type="submit" disabled={addingStudent || apiStatus !== "online" || enrollmentPhotos.length !== 30}>
            <Users size={16}/> {addingStudent ? "Uploading & registering…" : enrollmentPhotos.length === 30 ? "Register student with 30 images" : `Capture ${30 - enrollmentPhotos.length} more photo(s)`}
          </button>
          {studentMessage && <p className="formMessage" role="status">{studentMessage}</p>}
        </form>
      </section>

      <section className="studentsCard">
        <div className="sectionHeading"><div><strong>Registered students</strong><small>Active students · archived students retain attendance history</small></div><span className="countBadge">{students.length} total</span></div>
        {students.length === 0 ? <p className="emptyState">No students returned by the API. Register students through the backend first.</p> :
          <div className="tableWrap"><table><thead><tr><th>Student ID</th><th>Full name</th><th>Department</th><th>Section</th><th>Email</th><th>Action</th></tr></thead>
            <tbody>{students.map(student => <tr key={student.id}>
              <td>{student.student_id}</td><td>{student.full_name}</td><td>{student.department || "—"}</td>
              <td>{student.section || "—"}</td><td>{student.email}</td>
              <td><button className="deleteStudentButton" type="button" onClick={() => void deleteStudent(student)} disabled={apiStatus !== "online"} title={`Permanently delete ${student.full_name}`}><Trash2 size={14}/> Delete</button></td>
            </tr>)}</tbody></table></div>}
      </section>

      <footer>Built for CSE471 · Local development · Obtain consent and protect biometric data</footer>
      <canvas ref={canvasRef} className="hiddenCanvas" />
    </main>
  );
}
