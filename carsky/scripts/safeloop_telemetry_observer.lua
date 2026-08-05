-- SafeLoop telemetry observer for a CarSky Script Node.
-- Read-only: subscribes to the three ego signals published by TripReplayer.

local kuksa = pins.kuksa
assert(kuksa and kuksa.vss, "pins.kuksa.vss missing")

local PATHS = {
    "Vehicle.Speed",
    "Vehicle.Acceleration.Longitudinal",
    "Vehicle.Acceleration.Lateral",
}

local last_value = {}

kuksa:subscribe(PATHS)
kuksa:on_change(function(event)
    if last_value[event.path] ~= event.value then
        last_value[event.path] = event.value
        log(string.format(
            "[safeloop.telemetry.v1] %s=%s",
            event.path,
            tostring(event.value)
        ))
    end
end)

log("[safeloop.telemetry.v1] observer ready; paths=3; mode=read-only")
