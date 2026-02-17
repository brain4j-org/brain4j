const lossBtn = document.getElementById("lossBtn");
const accuracyBtn = document.getElementById("accuracyBtn");

lossBtn.addEventListener("click", () => {
    lossBtn.classList.add("active");
    accuracyBtn.classList.remove("active");
    console.log("Switch to Loss");
    // TODO: update chart dataset
});

accuracyBtn.addEventListener("click", () => {
    accuracyBtn.classList.add("active");
    lossBtn.classList.remove("active");
    console.log("Switch to Accuracy");
    // TODO: update chart dataset
});