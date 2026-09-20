const button = document.querySelector("#screenButton");
const response = document.querySelector("#response");
const status = document.querySelector("#status");
const details = document.querySelector("#details");

button.addEventListener("click", async () => {
  button.disabled = true;
  button.innerHTML = "Capturing…";
  status.textContent = "Working locally";
  response.textContent = "Capturing one screen image and checking it for readable text…";
  details.hidden = true;

  try {
    const request = await fetch("/api/screen", { method: "POST" });
    const data = await request.json();
    if (!request.ok) throw new Error(data.detail || "Screen analysis failed.");

    response.textContent = data.response;
    status.textContent = data.status === "success" ? "Complete" : "Needs setup";
    document.querySelector("#context").textContent = data.visual_context;
    document.querySelector("#screenText").textContent = data.screen_text || "No readable text returned.";
    document.querySelector("#note").textContent = data.note || "";
    details.hidden = false;
  } catch (error) {
    status.textContent = "Error";
    response.textContent = error.message;
  } finally {
    button.disabled = false;
    button.innerHTML = "Analyze screen <span>→</span>";
  }
});
