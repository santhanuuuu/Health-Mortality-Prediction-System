/**
 * Global Health & Mortality Prediction System
 * Client-side validation and loading animation
 */

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictionForm");
    const submitBtn = document.getElementById("submitBtn");

    if (!form || !submitBtn) return;

    // Create loading overlay
    const overlay = document.createElement("div");
    overlay.className = "loading-overlay";
    overlay.id = "loadingOverlay";
    overlay.innerHTML =
        '<div class="loading-spinner">' +
        '<div class="spinner-border text-light mb-3" role="status"></div>' +
        "<h5>Analyzing Your Health Survey...</h5>" +
        '<p class="text-white-50 mb-0">Generating your personal assessment</p>' +
        "</div>";
    document.body.appendChild(overlay);

    const btnText = submitBtn.querySelector(".btn-text");
    const btnLoader = submitBtn.querySelector(".btn-loader");

    form.addEventListener("submit", function (e) {
        const errors = validateForm(form);
        if (errors.length > 0) {
            e.preventDefault();
            showErrors(form, errors);
            return;
        }

        if (btnText) btnText.classList.add("d-none");
        if (btnLoader) btnLoader.classList.remove("d-none");
        submitBtn.disabled = true;
        overlay.classList.add("active");
    });

    form.querySelectorAll("input, select").forEach(function (field) {
        field.addEventListener("blur", function () {
            validateField(field);
        });
        field.addEventListener("input", function () {
            if (field.classList.contains("is-invalid")) {
                validateField(field);
            }
        });
    });

    const navbar = document.querySelector(".custom-navbar");
    if (navbar) {
        window.addEventListener("scroll", function () {
            if (window.scrollY > 50) {
                navbar.style.padding = "0.4rem 0";
            } else {
                navbar.style.padding = "";
            }
        });
    }

    const observer = new IntersectionObserver(
        function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add("animate-slide-up");
                }
            });
        },
        { threshold: 0.2 }
    );

    document.querySelectorAll(".stat-card").forEach(function (card) {
        observer.observe(card);
    });

    // Sleep hours slider label
    const sleepSlider = document.getElementById("sleep_hours");
    const sleepLabel = document.getElementById("sleepHoursValue");
    if (sleepSlider && sleepLabel) {
        sleepLabel.textContent = sleepSlider.value;
        sleepSlider.addEventListener("input", function () {
            sleepLabel.textContent = sleepSlider.value;
        });
    }
});

function validateForm(form) {
    const errors = [];
    form.querySelectorAll("[required]").forEach(function (field) {
        const fieldErrors = getFieldErrors(field);
        if (fieldErrors.length > 0) {
            errors.push(fieldErrors[0]);
        }
    });
    return errors;
}

function validateField(field) {
    const errors = getFieldErrors(field);
    if (errors.length > 0) {
        field.classList.add("is-invalid");
        return false;
    }
    field.classList.remove("is-invalid");
    return true;
}

function getFieldErrors(field) {
    const errors = [];
    const label =
        field.labels && field.labels[0]
            ? field.labels[0].textContent
            : "Field";
    const value = field.value.trim();

    if (field.hasAttribute("required") && !value) {
        errors.push(label + " is required.");
        return errors;
    }

    if (field.type === "number" && value !== "") {
        const num = parseFloat(value);
        const min = field.min !== "" ? parseFloat(field.min) : null;
        const max = field.max !== "" ? parseFloat(field.max) : null;

        if (isNaN(num)) {
            errors.push(label + " must be a valid number.");
        } else {
            if (min !== null && num < min) {
                errors.push(label + " must be at least " + min + ".");
            }
            if (max !== null && num > max) {
                errors.push(label + " must be at most " + max + ".");
            }
        }
    }

    return errors;
}

function showErrors(form, errors) {
    form.querySelectorAll("[required]").forEach(validateField);

    if (errors.length > 0) {
        const alertDiv = document.createElement("div");
        alertDiv.className =
            "alert alert-danger alert-dismissible fade show mt-3";
        alertDiv.setAttribute("role", "alert");
        alertDiv.innerHTML =
            "<strong>Please fix the following:</strong><ul class=\"mb-0 mt-2\">" +
            errors
                .map(function (e) {
                    return "<li>" + e + "</li>";
                })
                .join("") +
            '</ul><button type="button" class="btn-close" data-bs-dismiss="alert"></button>';

        const existing = form.querySelector(".alert-danger");
        if (existing) existing.remove();

        form.insertBefore(alertDiv, form.firstChild);
        alertDiv.scrollIntoView({ behavior: "smooth", block: "center" });
    }
}
