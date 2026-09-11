const inventory = [
  {
    breakfastIncluded: true,
    id: "luma_harbor",
    name: "Luma Harbor Hotel",
    neighborhood: "Marina District",
    rateLabel: "$289 / night",
    summary: "Harbor-facing suites with breakfast service and a writing desk.",
    workspaceDesk: true,
  },
  {
    breakfastIncluded: false,
    id: "ember_lane",
    name: "Ember Lane Hotel",
    neighborhood: "Mission Bay",
    rateLabel: "$241 / night",
    summary: "Modern micro-hotel near the waterfront with compact rooms.",
    workspaceDesk: true,
  },
  {
    breakfastIncluded: true,
    id: "port_exchange",
    name: "Port Exchange Inn",
    neighborhood: "Embarcadero",
    rateLabel: "$264 / night",
    summary: "Historic property with breakfast service but no in-room desk.",
    workspaceDesk: false,
  },
];

const visualsByHotelId = {
  ember_lane: "resultVisualEmber",
  luma_harbor: "resultVisualLuma",
  port_exchange: "resultVisualPort",
};

const state = {
  confirmation: null,
  filters: {
    neighborhood: "",
    requireBreakfast: false,
    requireWorkspace: false,
  },
  form: {
    checkIn: "",
    checkOut: "",
    guestEmail: "",
    guestName: "",
    specialRequest: "",
  },
  selectedHotelId: null,
  visibleHotelIds: inventory.map((hotel) => hotel.id),
};

const filterNeighborhood = document.querySelector("[data-testid='filter-neighborhood']");
const filterBreakfast = document.querySelector("[data-testid='filter-breakfast']");
const filterWorkspace = document.querySelector("[data-testid='filter-workspace']");
const applyFiltersButton = document.querySelector("[data-testid='apply-search-filters']");
const resultsListElement = document.querySelector("[data-testid='results-list']");
const selectedStayElement = document.querySelector("[data-testid='selected-stay']");
const guestNameInput = document.querySelector("[data-testid='guest-name']");
const guestEmailInput = document.querySelector("[data-testid='guest-email']");
const checkInInput = document.querySelector("[data-testid='check-in']");
const checkOutInput = document.querySelector("[data-testid='check-out']");
const specialRequestInput = document.querySelector("[data-testid='special-request']");
const confirmButton = document.querySelector("[data-testid='confirm-reservation']");
const bookingStatusElement = document.querySelector("[data-testid='booking-status']");
const filterStatusElement = document.querySelector("[data-testid='filter-status']");
const confirmationHotelElement = document.querySelector("[data-testid='confirmation-hotel']");
const confirmationGuestElement = document.querySelector("[data-testid='confirmation-guest']");
const confirmationDatesElement = document.querySelector("[data-testid='confirmation-dates']");
const confirmationRequestElement = document.querySelector("[data-testid='confirmation-request']");
const bookingStatePillElement = document.querySelector("[data-role='booking-state-pill']");
const confirmationTitleElement = document.querySelector("[data-role='confirmation-title']");
const confirmationCopyElement = document.querySelector("[data-role='confirmation-copy']");

function readConfirmation() {
  return state.confirmation ? { ...state.confirmation } : null;
}

function readFilters() {
  return { ...state.filters };
}

function selectedHotel() {
  return inventory.find((hotel) => hotel.id === state.selectedHotelId) ?? null;
}

function formatVisibleResultCount(count) {
  return `${count} ${count === 1 ? "stay" : "stays"}`;
}

function applyFilters() {
  state.filters = {
    neighborhood: filterNeighborhood.value,
    requireBreakfast: filterBreakfast.checked,
    requireWorkspace: filterWorkspace.checked,
  };

  state.visibleHotelIds = inventory
    .filter((hotel) => {
      if (state.filters.neighborhood && hotel.neighborhood !== state.filters.neighborhood) {
        return false;
      }

      if (state.filters.requireBreakfast && !hotel.breakfastIncluded) {
        return false;
      }

      if (state.filters.requireWorkspace && !hotel.workspaceDesk) {
        return false;
      }

      return true;
    })
    .map((hotel) => hotel.id);

  if (state.selectedHotelId && !state.visibleHotelIds.includes(state.selectedHotelId)) {
    state.selectedHotelId = null;
  }

  render();
}

function selectHotel(hotelId) {
  state.selectedHotelId = hotelId;
  render();
}

function updateForm() {
  state.form = {
    checkIn: checkInInput.value,
    checkOut: checkOutInput.value,
    guestEmail: guestEmailInput.value,
    guestName: guestNameInput.value,
    specialRequest: specialRequestInput.value,
  };
}

function confirmReservation() {
  updateForm();
  const hotel = selectedHotel();

  if (!hotel) {
    return;
  }

  state.confirmation = {
    checkIn: state.form.checkIn,
    checkOut: state.form.checkOut,
    guestEmail: state.form.guestEmail,
    guestName: state.form.guestName,
    hotelId: hotel.id,
    hotelName: hotel.name,
    specialRequest: state.form.specialRequest,
  };

  render();
}

function renderResults() {
  resultsListElement.innerHTML = "";

  const visibleHotels = inventory.filter((hotel) => state.visibleHotelIds.includes(hotel.id));
  filterStatusElement.textContent = formatVisibleResultCount(visibleHotels.length);

  for (const hotel of visibleHotels) {
    const article = document.createElement("article");
    article.className = "resultCard";
    article.dataset.testid = `hotel-${hotel.id}`;

    article.innerHTML = `
      <div class="resultVisual ${visualsByHotelId[hotel.id]}"></div>
      <div class="resultBody">
        <div class="resultTitleBlock">
          <strong>${hotel.name}</strong>
          <p class="resultLocation">${hotel.neighborhood}</p>
        </div>
        <p class="resultSummary">${hotel.summary}</p>
        <div class="badgeRow">
          <span class="badge">${hotel.breakfastIncluded ? "Breakfast included" : "No breakfast"}</span>
          <span class="badge">${hotel.workspaceDesk ? "Workspace desk" : "No desk"}</span>
        </div>
      </div>
      <div class="resultRateColumn">
        <div>
          <div class="resultRateLabel">Member rate</div>
          <div class="resultRate">${hotel.rateLabel}</div>
        </div>
        <span class="resultMatch">${
          state.selectedHotelId === hotel.id ? "Selected stay" : "Available now"
        }</span>
        <div>
          <button
            class="hotelSelectButton${state.selectedHotelId === hotel.id ? " isSelected" : ""}"
            type="button"
            data-testid="hotel-${hotel.id}-select"
          >
            ${state.selectedHotelId === hotel.id ? "Selected" : "Choose stay"}
          </button>
        </div>
      </div>
    `;

    article
      .querySelector(`[data-testid='hotel-${hotel.id}-select']`)
      .addEventListener("click", () => selectHotel(hotel.id));

    resultsListElement.append(article);
  }
}

function renderForm() {
  const hotel = selectedHotel();

  if (!hotel) {
    selectedStayElement.innerHTML = `
      <strong>Select a stay</strong>
      <p>Choose a hotel to review details and continue.</p>
    `;
  } else {
    selectedStayElement.innerHTML = `
      <strong>${hotel.name}</strong>
      <p>${hotel.neighborhood} · ${hotel.rateLabel}</p>
    `;
  }

  guestNameInput.value = state.form.guestName;
  guestEmailInput.value = state.form.guestEmail;
  checkInInput.value = state.form.checkIn;
  checkOutInput.value = state.form.checkOut;
  specialRequestInput.value = state.form.specialRequest;

  confirmButton.disabled = !hotel;
}

function renderConfirmation() {
  const confirmation = state.confirmation;

  if (!confirmation) {
    confirmationHotelElement.textContent = "none";
    confirmationGuestElement.textContent = "none";
    confirmationDatesElement.textContent = "none";
    confirmationRequestElement.textContent = "none";
    confirmationTitleElement.textContent = "No reservation yet";
    confirmationCopyElement.textContent =
      "Your confirmed stay will appear here once the reservation is submitted.";
    return;
  }

  confirmationHotelElement.textContent = confirmation.hotelName;
  confirmationGuestElement.textContent = `${confirmation.guestName} · ${confirmation.guestEmail}`;
  confirmationDatesElement.textContent = `${confirmation.checkIn} → ${confirmation.checkOut}`;
  confirmationRequestElement.textContent = confirmation.specialRequest;
  confirmationTitleElement.textContent = `${confirmation.hotelName} confirmed`;
  confirmationCopyElement.textContent = `${confirmation.checkIn} to ${confirmation.checkOut} · ${confirmation.guestName}`;
}

function renderStatus() {
  const confirmed = state.confirmation !== null;

  bookingStatusElement.textContent = confirmed
    ? "Reservation recorded"
    : "Reservation pending";

  bookingStatePillElement.textContent = confirmed ? "Confirmed" : "Pending";
  bookingStatePillElement.classList.toggle("isConfirmed", confirmed);
}

function render() {
  renderResults();
  renderForm();
  renderConfirmation();
  renderStatus();
}

applyFiltersButton.addEventListener("click", applyFilters);
guestNameInput.addEventListener("input", updateForm);
guestEmailInput.addEventListener("input", updateForm);
checkInInput.addEventListener("input", updateForm);
checkOutInput.addEventListener("input", updateForm);
specialRequestInput.addEventListener("input", updateForm);
confirmButton.addEventListener("click", confirmReservation);

window.__bookingLabReady = true;
window.__bookingReadConfirmation = readConfirmation;
window.__bookingReadFilters = readFilters;

render();
