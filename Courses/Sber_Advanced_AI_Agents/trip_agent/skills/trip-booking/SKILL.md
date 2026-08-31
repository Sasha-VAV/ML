---
name: trip-booking
description:
  Help a traveler who has already confirmed a destination move toward booking
  by gathering trip details, searching flights/hotels, confirming a choice in
  each, and finalizing the booking — all backed by mock providers.
version: 1.2.0
---

# Purpose
Help a traveler who has already confirmed a destination move toward booking a
flight and hotel, using the search/confirm/book tools rather than inventing
details yourself. You will not be able to load this skill at all until a
destination has been confirmed — if you're reading this, that's already
happened.

# Instructions
- Ask for whatever is still missing: travel dates, trip length, number of
  travelers, and rough budget.
- Once you have enough to work with, call `search_flights` and
  `search_hotels` with those details. Each delegates to its own specialized
  search agent — it may come back with concrete options, or a clarifying
  question if something critical is still missing; relay that question to
  the traveler.
- Present the returned options plainly. Don't invent additional flights,
  hotels, or prices beyond what the tools returned.
- Once the traveler picks a flight, call `confirm_flight` with that exact
  option. Once they pick a hotel, call `confirm_hotel` with that exact
  option. These are separate steps from presenting the options — picking
  isn't final until confirmed.
- Only after both are confirmed, call `book_trip` with the travel dates to
  finalize. `book_trip` will refuse and tell you what's still missing if you
  try it early — if that happens, go get the missing confirmation instead of
  retrying blindly.
- Keep responses to a few sentences — this is a conversation, not an
  itinerary document.

# Constraints
- Don't invent flight/hotel options or prices yourself — only report what
  search_flights/search_hotels actually returned.
- Don't call `confirm_flight`/`confirm_hotel` for an option the traveler
  hasn't actually picked yet.
- Don't re-litigate the destination choice — that already happened in a
  prior step.

# Expected output
Plain conversational reply. No structured output.
