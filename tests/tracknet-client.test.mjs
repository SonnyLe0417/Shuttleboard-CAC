import test from "node:test";
import assert from "node:assert/strict";
import {
  cancelTrackNetJob,
  getTrackNetHealth,
  getTrackNetResult,
  submitTrackNetVideo,
  waitForTrackNet,
} from "../src/lib/tracknet-client.ts";

function withFetch(t, fetch, origin) {
  const realFetch = globalThis.fetch;
  globalThis.fetch = fetch;
  if (origin) globalThis.location = { origin };
  t.after(() => {
    globalThis.fetch = realFetch;
    delete globalThis.location;
  });
}

test("a TrackNet service that cannot be reached is explained, not just 'Failed to fetch'", async (t) => {
  // A stopped service and a service refusing the page both surface this way.
  withFetch(
    t,
    async () => {
      throw new TypeError("Failed to fetch");
    },
    "http://127.0.0.1:5175",
  );
  await assert.rejects(submitTrackNetVideo(new File([new Uint8Array(4)], "match.mp4")), (error) => {
    assert.match(
      error.message,
      /Could not reach TrackNet at http:\/\/127\.0\.0\.1:8000 from this page \(http:\/\/127\.0\.0\.1:5175\)/,
    );
    assert.match(error.message, /restart it/);
    assert.match(error.message, /V3 engine, other addresses go in TRACKNET_ALLOW_ORIGINS/);
    assert.ok(error.cause instanceof TypeError);
    return true;
  });
  // Cancelling against a stopped service stays quiet, so a new video can be chosen.
  await cancelTrackNetJob("a".repeat(32));
});

test("a missing job is told apart by its status, not by numbers in a message", async (t) => {
  withFetch(t, async () => new Response('{"detail":"Unknown job"}', { status: 404 }));
  await assert.rejects(getTrackNetResult("a".repeat(32)), { status: 404 });
  await assert.rejects(waitForTrackNet("a".repeat(32)), { status: 404 });
  // A page on port 4040 that cannot reach the service is not a missing job.
  withFetch(
    t,
    async () => {
      throw new TypeError("Failed to fetch");
    },
    "http://127.0.0.1:4040",
  );
  await assert.rejects(waitForTrackNet("a".repeat(32)), (error) => {
    assert.match(error.message, /4040/);
    assert.equal(error.status, undefined);
    return true;
  });
});

test("a cancelled TrackNet request is still a cancellation", async (t) => {
  withFetch(t, async () => {
    throw new DOMException("Cancelled", "AbortError");
  });
  await assert.rejects(getTrackNetHealth(), { name: "AbortError" });
});
