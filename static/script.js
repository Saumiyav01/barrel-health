let mainChart;
let stressStrainChart;
let healthGauge;
let interval;

async function initializeSystem() {

    const cycles = parseInt(document.getElementById("initCycles").value);

    await fetch("/initialize", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ cycles: cycles })
    });

    document.getElementById("initModal").style.display = "none";

    initializeCharts();
    startMonitoring();
}

function initializeCharts() {

    mainChart = new Chart(document.getElementById("mainChart"), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: "Temp (K)", data: [], borderColor: "#f97316", tension:0.3 },
                { label: "Acceleration (m/s²)", data: [], borderColor: "#22d3ee", tension:0.3 }
            ]
        },
        options: { responsive:true, maintainAspectRatio:false }
    });

    stressStrainChart = new Chart(
        document.getElementById("stressStrainChart"),
        {
            type: 'scatter',
            data: {
                datasets: [{
                    label: "Stress–Strain",
                    data: [],
                    borderColor: "#22c55e",
                    backgroundColor: "#22c55e",
                    showLine: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: "Strain" }},
                    y: { title: { display: true, text: "Stress (MPa)" }}
                }
            }
        }
    );

    healthGauge = new Chart(document.getElementById("healthGauge"), {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [100,0],
                backgroundColor:["#22c55e","#1f2937"],
                borderWidth:0
            }]
        },
        options:{
            cutout:"70%",
            plugins:{legend:{display:false}}
        }
    });
}

function startMonitoring() {
    interval = setInterval(fetchData, 500);
}

async function fetchData() {

    const res = await fetch("/latest");
    const data = await res.json();

    document.getElementById("temp").innerText = data.temp.toFixed(2);
    document.getElementById("vibration").innerText = data.vibration.toFixed(3);
    document.getElementById("strain").innerText = data.strain.toFixed(7);

    updateCharts(data);
    updateGauge(data);
    updateTable(data);

    if(data.status === "Expired"){
        clearInterval(interval);
        alert("Barrel Expired");
    }
}

function updateCharts(data){

    mainChart.data.labels.push(data.cycles);
    mainChart.data.datasets[0].data.push(data.temp);
    mainChart.data.datasets[1].data.push(data.vibration);

    if(mainChart.data.labels.length > 100){
        mainChart.data.labels.shift();
        mainChart.data.datasets.forEach(ds=>ds.data.shift());
    }

    mainChart.update();

    stressStrainChart.data.datasets[0].data.push({
        x: data.strain,
        y: data.stress
    });

    if(stressStrainChart.data.datasets[0].data.length > 100){
        stressStrainChart.data.datasets[0].data.shift();
    }

    stressStrainChart.update();
}

function updateGauge(data){

    const hp = data.health * 100;

    healthGauge.data.datasets[0].data = [hp,100-hp];

    if(hp > 60)
        healthGauge.data.datasets[0].backgroundColor[0]="#22c55e";
    else if(hp > 30)
        healthGauge.data.datasets[0].backgroundColor[0]="#f59e0b";
    else
        healthGauge.data.datasets[0].backgroundColor[0]="#ef4444";

    healthGauge.update();
}

function updateTable(data){

    const table = document.getElementById("cycleHistory");

    const row = document.createElement("tr");

    row.innerHTML = `
        <td>${data.cycles}</td>
        <td>${data.temp.toFixed(1)}</td>
        <td>${data.vibration.toFixed(2)}</td>
        <td>${(data.health*100).toFixed(1)}%</td>
        <td>${data.status}</td>
    `;

    table.prepend(row);

    if(table.rows.length > 5)
        table.deleteRow(5);
}