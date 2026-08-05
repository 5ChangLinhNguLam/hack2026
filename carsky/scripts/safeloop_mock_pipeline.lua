-- SafeLoop MOCK C1 + C2 + Risk Fusion for a CarSky Script Node.
--
-- Purpose: wire and demonstrate the end-to-end product before trained models
-- land.  This script is deterministic, marked MOCK in logs, and never reads
-- ground truth.  It publishes sensor simulation only; it does not actuate a
-- real brake, hazard light, or other vehicle control.

local kuksa = pins.kuksa
assert(kuksa and kuksa.vss, "pins.kuksa.vss missing")

local Vehicle = kuksa.vss.Vehicle
local obstacle = Vehicle.ADAS.ObstacleDetection.Front.Center
local driver = Vehicle.Driver
local dms = Vehicle.ADAS.DMS

local PERIOD_MS = 50
local SCENARIO_MS = 16000
local frame_id = 0
local last_state = nil
local last_action = nil
local last_scenario_t_ms = nil
local c3_collision_penalty = 0.0
local c3_driver_penalty = 0.0
local c3_comfort_penalty = 0.0

local function lerp(a, b, ratio)
    local r = math.max(0.0, math.min(1.0, ratio))
    return a + (b - a) * r
end

local function mock_ttc(t_ms)
    if t_ms < 4000 then return lerp(8.0, 5.0, t_ms / 4000) end
    if t_ms < 8000 then return lerp(5.0, 2.2, (t_ms - 4000) / 4000) end
    if t_ms < 12000 then return lerp(2.2, 0.8, (t_ms - 8000) / 4000) end
    return lerp(0.8, 8.0, (t_ms - 12000) / 4000)
end

local function mock_driver(t_ms)
    if t_ms < 4000 then return "alert", 95.0, 5.0, 5.0, true end
    if t_ms < 8000 then return "distracted", 32.0, 92.0, 20.0, false end
    if t_ms < 10000 then return "drowsy", 43.0, 12.0, 78.0, true end
    if t_ms < 12000 then return "microsleep", 5.0, 8.0, 98.0, false end
    if t_ms < 14000 then return "yawning", 55.0, 10.0, 68.0, true end
    return "alert", 95.0, 5.0, 5.0, true
end

local function collision_risk(ttc)
    if ttc <= 1.0 then return 1.0 end
    if ttc <= 2.0 then return lerp(1.0, 0.80, ttc - 1.0) end
    if ttc <= 3.0 then return lerp(0.80, 0.55, ttc - 2.0) end
    if ttc <= 5.0 then return lerp(0.55, 0.20, (ttc - 3.0) / 2.0) end
    return lerp(0.20, 0.05, (ttc - 5.0) / 3.0)
end

local function risk_action(ttc, attentive, distraction, fatigue)
    local driver_risk = math.max(100.0 - attentive, distraction, fatigue) / 100.0
    local score = math.min(1.0, 0.75 * collision_risk(ttc) + 0.45 * driver_risk)
    if ttc <= 1.2 or (ttc < 2.0 and driver_risk >= 0.70) then
        return score, "CRITICAL", "EMERGENCY_BRAKE_REQUEST"
    end
    if ttc < 2.5 or score >= 0.75 then
        return score, "HIGH", "VISUAL_AUDIO_HAPTIC_WARNING"
    end
    if score >= 0.45 then return score, "CAUTION", "VISUAL_WARNING" end
    return score, "SAFE", "MONITOR"
end

local function c3_grade(score)
    if score >= 90 then return "A" end
    if score >= 80 then return "B" end
    if score >= 70 then return "C" end
    if score >= 60 then return "D" end
    return "E"
end

local function update_c3(t_ms, ttc, attentive, distraction, fatigue, longitudinal, lateral)
    if last_scenario_t_ms ~= nil and t_ms < last_scenario_t_ms then
        c3_collision_penalty = 0.0
        c3_driver_penalty = 0.0
        c3_comfort_penalty = 0.0
    end
    last_scenario_t_ms = t_ms

    local collision_exposure = math.max(0.0, math.min(1.0, (3.0 - ttc) / 2.2))
    local driver_exposure = math.max(100.0 - attentive, distraction, fatigue) / 100.0
    local comfort_exposure = math.min(1.0,
        math.max(0.0, math.abs(longitudinal) - 2.0) / 6.0
        + math.max(0.0, math.abs(lateral) - 1.5) / 4.0)

    c3_collision_penalty = c3_collision_penalty + 0.05 * 2.8 * collision_exposure
    c3_driver_penalty = c3_driver_penalty + 0.05 * 1.5 * driver_exposure
    c3_comfort_penalty = c3_comfort_penalty + 0.05 * 0.7 * comfort_exposure
    local penalty = c3_collision_penalty + c3_driver_penalty + c3_comfort_penalty
    local score = math.max(0.0, 100.0 - penalty)
    return score, c3_grade(score)
end

timer.periodic(PERIOD_MS, function()
    local t_ms = (frame_id * PERIOD_MS) % SCENARIO_MS
    local angle = 2.0 * math.pi * t_ms / SCENARIO_MS
    local speed = 45.0 + 5.0 * math.sin(angle)
    local longitudinal = 0.8 * math.cos(angle)
    local lateral = 0.25 * math.sin(2.0 * angle)

    local ttc = mock_ttc(t_ms)
    local distance = math.max(0.5, (speed / 3.6) * ttc)
    local state, attentive, distraction, fatigue, eyes_on_road = mock_driver(t_ms)
    local score, level, action = risk_action(
        ttc, attentive, distraction, fatigue
    )
    local c3_score, c3_grade_value = update_c3(
        t_ms, ttc, attentive, distraction, fatigue, longitudinal, lateral
    )

    Vehicle.Speed:publish(speed)
    Vehicle.Acceleration.Longitudinal:publish(longitudinal)
    Vehicle.Acceleration.Lateral:publish(lateral)

    -- TimeGap is uint32 milliseconds in the starter VSS artifact.
    obstacle.TimeGap:publish(math.floor(ttc * 1000.0 + 0.5))
    obstacle.Distance:publish(distance)
    obstacle.IsWarning:publish(ttc < 2.0)

    driver.AttentiveProbability:publish(attentive)
    driver.DistractionLevel:publish(distraction)
    driver.FatigueLevel:publish(fatigue)
    driver.IsEyesOnRoad:publish(eyes_on_road)
    dms.IsWarning:publish(state ~= "alert")

    if state ~= last_state or action ~= last_action or frame_id % 20 == 0 then
        log(string.format(
            "[SafeLoop MOCK] frame=%d ttc=%.2fs driver=%s risk=%.2f c3=%.1f/%s level=%s action=%s",
            frame_id, ttc, state, score, c3_score, c3_grade_value, level, action
        ))
        last_state = state
        last_action = action
    end

    frame_id = frame_id + 1
end)

log("[SafeLoop MOCK] ready; C1+C2+C3+Fusion; period_ms=50; scenario_ms=16000; actuator=false")
