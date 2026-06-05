const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyzeButton = document.getElementById("analyzeButton");

const mainResult = document.getElementById("mainResult");
const predictions = document.getElementById("predictions");

let selectedFile = null;

imageInput.addEventListener("change", (event) => {

    selectedFile = event.target.files[0];

    if (!selectedFile) return;

    preview.src = URL.createObjectURL(selectedFile);

    preview.style.display = "block";

    mainResult.innerHTML = "";
    predictions.innerHTML = "";
});

analyzeButton.addEventListener("click", async () => {

    if (!selectedFile) {
        alert("Selecione uma imagem.");
        return;
    }

    mainResult.innerHTML = "<h2>Processando...</h2>";
    predictions.innerHTML = "";

    const formData = new FormData();

    formData.append("image", selectedFile);

    try {

        const response = await fetch("/infer", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!data.ok) {
            throw new Error(data.error);
        }

        renderResult(data);

    } catch (error) {

        mainResult.innerHTML = `
            <h2>Erro</h2>
            <p>${error.message}</p>
        `;
    }
});

function renderResult(data) {

    mainResult.innerHTML = `
        <div class="result-class">
            ${data.predictedClass}
        </div>

        <div class="confidence">
            Confiança:
            ${(data.confidence * 100).toFixed(2)}%
        </div>

        <hr>
    `;

    predictions.innerHTML = "";

    data.topPredictions.forEach(prediction => {

        const percent =
            (prediction.confidence * 100).toFixed(2);

        const row = document.createElement("div");

        row.className = "prediction-item";

        row.innerHTML = `
            <div class="prediction-header">
                <span>${prediction.class}</span>
                <span>${percent}%</span>
            </div>

            <div class="bar-container">
                <div
                    class="bar"
                    style="width:${percent}%"
                ></div>
            </div>
        `;

        predictions.appendChild(row);
    });
}