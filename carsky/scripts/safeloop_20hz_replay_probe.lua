-- SafeLoop deterministic 20 Hz replay probe for a CarSky Script Node.
--
-- This is an integration probe, not the production trip data plane. It emits
-- 20 consecutive T01-Sample ego frames through the in-room KUKSA connection,
-- then neutralizes the three signals exactly once. Subsequent timer callbacks
-- are no-ops so the probe cannot loop or leave a periodic publisher running.

local kuksa = pins.kuksa
assert(kuksa and kuksa.vss, "pins.kuksa.vss missing")

local Vehicle = kuksa.vss.Vehicle
local PERIOD_MS = 50
local frames = {
    { frame_id = 10, speed = 0.55, longitudinal = 2.717, lateral = -0.029 },
    { frame_id = 11, speed = 3.38, longitudinal = 16.047, lateral = -0.2 },
    { frame_id = 12, speed = 5.81, longitudinal = 13.513, lateral = -0.145 },
    { frame_id = 13, speed = 7.7, longitudinal = 10.507, lateral = -0.127 },
    { frame_id = 14, speed = 9.19, longitudinal = 8.24, lateral = -0.091 },
    { frame_id = 15, speed = 10.52, longitudinal = 7.405, lateral = -0.099 },
    { frame_id = 16, speed = 11.8, longitudinal = 7.105, lateral = 0.283 },
    { frame_id = 17, speed = 12.97, longitudinal = 6.537, lateral = 0.054 },
    { frame_id = 18, speed = 14.07, longitudinal = 6.087, lateral = -0.053 },
    { frame_id = 19, speed = 15.11, longitudinal = 5.807, lateral = -0.02 },
    { frame_id = 20, speed = 16.13, longitudinal = 5.635, lateral = -0.036 },
    { frame_id = 21, speed = 17.12, longitudinal = 5.525, lateral = -0.016 },
    { frame_id = 22, speed = 18.1, longitudinal = 5.441, lateral = -0.02 },
    { frame_id = 23, speed = 19.04, longitudinal = 5.231, lateral = -0.019 },
    { frame_id = 24, speed = 19.36, longitudinal = 1.753, lateral = -0.022 },
    { frame_id = 25, speed = 19.62, longitudinal = 1.462, lateral = -0.004 },
    { frame_id = 26, speed = 20.13, longitudinal = 2.834, lateral = -0.013 },
    { frame_id = 27, speed = 20.77, longitudinal = 3.537, lateral = -0.011 },
    { frame_id = 28, speed = 21.56, longitudinal = 4.37, lateral = -0.026 },
    { frame_id = 29, speed = 22.39, longitudinal = 4.652, lateral = 0.006 },
}

local next_index = 1
local neutralized = false

local function neutralize()
    Vehicle.Speed:publish(0.0)
    Vehicle.Acceleration.Longitudinal:publish(0.0)
    Vehicle.Acceleration.Lateral:publish(0.0)
    neutralized = true
    log("[safeloop.replay-probe] complete; frames=20; signals=60; neutralized=true")
end

timer.periodic(PERIOD_MS, function()
    if next_index <= #frames then
        local frame = frames[next_index]
        Vehicle.Speed:publish(frame.speed)
        Vehicle.Acceleration.Longitudinal:publish(frame.longitudinal)
        Vehicle.Acceleration.Lateral:publish(frame.lateral)
        log(string.format(
            "[safeloop.replay-probe] frame=%d index=%d/20",
            frame.frame_id,
            next_index
        ))
        next_index = next_index + 1
    elseif not neutralized then
        neutralize()
    end
end)

log("[safeloop.replay-probe] ready; frames=20; period_ms=50; target_hz=20")
