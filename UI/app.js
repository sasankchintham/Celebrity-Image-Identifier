const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const dropArea = document.getElementById("drop-area");
const resultTag = document.getElementById("result");

// 🔍 Preview Function
function showPreview(file) {
    const reader = new FileReader();
    reader.onload = () => {
        preview.src = reader.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(file);
}

// 🖱️ Handle File Selection
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        showPreview(file);
    }
});

// 🧲 Drag & Drop
dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");

    const file = e.dataTransfer.files[0];
    if (file) {
        fileInput.files = e.dataTransfer.files; // sync with input
        showPreview(file);
    }
});

// 🤖 Classify Button Click
document.getElementById("classify-btn").addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select an image.");
        return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
        const base64String = reader.result.split(",")[1]; // get pure base64

        try {
            const response = await fetch("http://127.0.0.1:5000/classify_image", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ image_data: base64String })
            });

            if (!response.ok) throw new Error("Network response was not ok");

            const data = await response.json();
            console.log("✅ Result:", data);

            if (data.predicted_class) {
                resultTag.textContent = `Predicted: ${data.predicted_class}`;
            } else {
                resultTag.textContent = data.error || "No result.";
            }
        } catch (err) {
            console.error("❌ Error:", err);
            resultTag.textContent = "Something went wrong.";
        }
    };

    reader.readAsDataURL(file);
});
