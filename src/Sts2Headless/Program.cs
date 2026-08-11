using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sts2Headless;

class Program
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    /// <summary>
    /// Locate the directory containing sts2.dll: STS2_LIB env, walk up from BaseDirectory, then BaseDirectory/lib.
    /// </summary>
    private static string ResolveLibDirectory()
    {
        var envLib = Environment.GetEnvironmentVariable("STS2_LIB");
        if (!string.IsNullOrWhiteSpace(envLib))
        {
            var p = Path.GetFullPath(envLib.Trim());
            if (Directory.Exists(p) && File.Exists(Path.Combine(p, "sts2.dll")))
                return p;
        }

        var dir = AppContext.BaseDirectory;
        for (var depth = 0; depth < 16 && !string.IsNullOrEmpty(dir); depth++)
        {
            var candidate = Path.Combine(dir, "lib");
            if (Directory.Exists(candidate) && File.Exists(Path.Combine(candidate, "sts2.dll")))
                return Path.GetFullPath(candidate);
            dir = Directory.GetParent(dir)?.FullName ?? "";
        }

        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "lib"));
    }

    static void Main(string[] args)
    {
        // Prevent unhandled exceptions from crashing the process
        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
        {
            Console.Error.WriteLine($"[FATAL] Unhandled: {e.ExceptionObject}");
        };
        TaskScheduler.UnobservedTaskException += (_, e) =>
        {
            Console.Error.WriteLine($"[WARN] Unobserved task exception: {e.Exception?.Message}");
            e.SetObserved();
        };

        var libDir = ResolveLibDirectory();

        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            var path = Path.Combine(libDir, name.Name + ".dll");
            if (File.Exists(path))
                return ctx.LoadFromAssemblyPath(Path.GetFullPath(path));

            // Also check game directory (via STS2_GAME_DIR env var)
            var gameDir = Environment.GetEnvironmentVariable("STS2_GAME_DIR") ?? "";
            if (!string.IsNullOrEmpty(gameDir))
            {
                path = Path.Combine(gameDir, name.Name + ".dll");
                if (File.Exists(path))
                    return ctx.LoadFromAssemblyPath(path);
            }

            return null;
        };

        var sim = new RunSimulator();
        WriteLine(new Dictionary<string, object?> { ["type"] = "ready", ["version"] = "0.2.0" });

        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line)) continue;

            Dictionary<string, object?>? result;
            try
            {
                var cmd = JsonSerializer.Deserialize<JsonElement>(line);
                result = HandleCommand(sim, cmd);
            }
            catch (JsonException ex)
            {
                result = new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Invalid JSON: {ex.Message}" };
            }
            catch (Exception ex)
            {
                result = new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"{ex.GetType().Name}: {ex.Message}" };
            }

            if (result != null)
            {
                WriteLine(result);
                if (result.TryGetValue("type", out var resultTypeObj) &&
                    string.Equals(resultTypeObj as string, "quit_result", StringComparison.Ordinal))
                {
                    break;
                }
            }
        }
    }

    static Dictionary<string, object?>? HandleCommand(RunSimulator sim, JsonElement cmd)
    {
        var cmdType = cmd.GetProperty("cmd").GetString() ?? "";
        switch (cmdType)
        {
            case "start_run":
                return sim.StartRun(
                    cmd.TryGetProperty("character", out var ch) ? ch.GetString() ?? "Ironclad" : "Ironclad",
                    cmd.TryGetProperty("ascension", out var asc) ? asc.GetInt32() : 0,
                    cmd.TryGetProperty("seed", out var s) ? s.GetString() : null,
                    cmd.TryGetProperty("lang", out var lang) ? lang.GetString() ?? "en" : "en"
                );

            case "action":
            {
                var action = cmd.GetProperty("action").GetString() ?? "";
                Dictionary<string, object?>? actionArgs = null;
                if (cmd.TryGetProperty("args", out var argsElem))
                {
                    actionArgs = new Dictionary<string, object?>();
                    foreach (var prop in argsElem.EnumerateObject())
                    {
                        actionArgs[prop.Name] = prop.Value.ValueKind switch
                        {
                            JsonValueKind.Number => prop.Value.GetInt32(),
                            JsonValueKind.String => prop.Value.GetString(),
                            JsonValueKind.True => true,
                            JsonValueKind.False => false,
                            _ => prop.Value.ToString(),
                        };
                    }
                }
                return sim.ExecuteAction(action, actionArgs);
            }

            case "load_save":
            {
                var savePath = cmd.TryGetProperty("path", out var sp) ? sp.GetString() : null;
                var saveJson = cmd.TryGetProperty("json", out var sj) ? sj.GetString() : null;
                if (saveJson == null && savePath != null)
                {
                    if (!File.Exists(savePath))
                        return new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Save file not found: {savePath}" };
                    saveJson = File.ReadAllText(savePath);
                }
                if (saveJson == null)
                    return new Dictionary<string, object?> { ["type"] = "error", ["message"] = "Provide 'path' or 'json' for load_save" };
                var loadLang = cmd.TryGetProperty("lang", out var le) ? (le.GetString() ?? "en") : "en";
                return sim.LoadSave(saveJson, loadLang);
            }
            case "get_map":
                return sim.GetFullMap();

            case "set_player":
            {
                var args = new Dictionary<string, JsonElement>();
                foreach (var prop in cmd.EnumerateObject())
                    if (prop.Name != "cmd") args[prop.Name] = prop.Value;
                return sim.SetPlayer(args);
            }

            case "enter_room":
            {
                var roomType = cmd.TryGetProperty("type", out var rt) ? rt.GetString() ?? "" : "";
                var encounter = cmd.TryGetProperty("encounter", out var enc) ? enc.GetString() : null;
                var eventId = cmd.TryGetProperty("event", out var ev) ? ev.GetString() : null;
                return sim.EnterRoom(roomType, encounter, eventId);
            }

            case "set_draw_order":
            {
                var cards = new List<string>();
                if (cmd.TryGetProperty("cards", out var cardsArr))
                    foreach (var c in cardsArr.EnumerateArray())
                        cards.Add(c.GetString() ?? "");
                return sim.SetDrawOrder(cards);
            }

            case "set_hand":
            {
                var cards = new List<string>();
                if (cmd.TryGetProperty("cards", out var handArr))
                    foreach (var c in handArr.EnumerateArray())
                        cards.Add(c.GetString() ?? "");
                List<string>? discard = null;
                if (cmd.TryGetProperty("discard", out var discArr))
                {
                    discard = new List<string>();
                    foreach (var c in discArr.EnumerateArray())
                        discard.Add(c.GetString() ?? "");
                }
                List<string>? exhaust = null;
                if (cmd.TryGetProperty("exhaust", out var exhArr))
                {
                    exhaust = new List<string>();
                    foreach (var c in exhArr.EnumerateArray())
                        exhaust.Add(c.GetString() ?? "");
                }
                return sim.SetHand(cards, discard, exhaust);
            }

            case "set_enemies":
            {
                var hps = new List<int>();
                if (cmd.TryGetProperty("hps", out var hpsArr))
                    foreach (var h in hpsArr.EnumerateArray())
                        hps.Add(h.GetInt32());
                List<int>? blocks = null;
                if (cmd.TryGetProperty("blocks", out var blkArr))
                {
                    blocks = new List<int>();
                    foreach (var b in blkArr.EnumerateArray())
                        blocks.Add(b.GetInt32());
                }
                return sim.SetEnemies(hps, blocks);
            }

            case "set_energy":
            {
                int energy = cmd.TryGetProperty("energy", out var enEl)
                    && enEl.ValueKind == System.Text.Json.JsonValueKind.Number
                    ? enEl.GetInt32() : 3;
                return sim.SetEnergy(energy);
            }

            case "set_powers":
            {
                static List<(string, int)> ParsePowers(System.Text.Json.JsonElement arr)
                {
                    var list = new List<(string, int)>();
                    foreach (var p in arr.EnumerateArray())
                    {
                        var id = p.TryGetProperty("id", out var idEl) ? idEl.GetString() : null;
                        var amt = p.TryGetProperty("amount", out var aEl)
                            && aEl.ValueKind == System.Text.Json.JsonValueKind.Number
                            ? aEl.GetInt32() : 0;
                        if (id != null) list.Add((id, amt));
                    }
                    return list;
                }
                var playerPowers = cmd.TryGetProperty("player", out var pArr)
                    ? ParsePowers(pArr) : new List<(string, int)>();
                var enemyPowers = new List<List<(string, int)>>();
                if (cmd.TryGetProperty("enemies", out var eArr))
                    foreach (var e in eArr.EnumerateArray())
                        enemyPowers.Add(ParsePowers(e));
                return sim.SetPowers(playerPowers, enemyPowers);
            }

            case "debug_members":
                return sim.DebugMembers();

            case "get_rng":
                return sim.GetRng();

            case "set_rng":
            {
                static ulong U(System.Text.Json.JsonElement o, string k)
                {
                    if (o.TryGetProperty(k, out var v)
                        && v.ValueKind == System.Text.Json.JsonValueKind.Number)
                    {
                        try { return v.GetUInt64(); }
                        catch { try { return (ulong)v.GetInt64(); } catch { return 0UL; } }
                    }
                    return 0UL;
                }
                static Dictionary<string, ulong[]> ParseRngs(System.Text.Json.JsonElement obj)
                {
                    var d = new Dictionary<string, ulong[]>();
                    foreach (var kv in obj.EnumerateObject())
                        d[kv.Name] = new[] { U(kv.Value, "s0"), U(kv.Value, "s1"),
                            U(kv.Value, "s2"), U(kv.Value, "s3"), U(kv.Value, "counter") };
                    return d;
                }
                var runRngs = cmd.TryGetProperty("run", out var rObj)
                    && rObj.ValueKind == System.Text.Json.JsonValueKind.Object
                    ? ParseRngs(rObj) : new Dictionary<string, ulong[]>();
                var playerRngs = cmd.TryGetProperty("player", out var plObj)
                    && plObj.ValueKind == System.Text.Json.JsonValueKind.Object
                    ? ParseRngs(plObj) : new Dictionary<string, ulong[]>();
                return sim.SetRng(runRngs, playerRngs);
            }

            case "write_continue_save":
            {
                var outputPath = cmd.TryGetProperty("path", out var op) ? op.GetString() : null;
                return sim.SaveCheckpoint(outputPath);
            }

            case "quit":
            {
                var outputPath = cmd.TryGetProperty("path", out var op) ? op.GetString() : null;
                if (!string.IsNullOrEmpty(outputPath))
                {
                    var saveResult = sim.SaveCheckpoint(outputPath);
                    bool saveOk = saveResult.TryGetValue("success", out var sObj) && sObj is bool b && b;
                    if (!saveOk)
                    {
                        // Save failed — do NOT clean up so the caller can retry with a different path.
                        return new Dictionary<string, object?>
                        {
                            ["type"] = "save_error",
                            ["save"] = saveResult,
                        };
                    }
                    sim.CleanUp();
                    return new Dictionary<string, object?>
                    {
                        ["type"] = "quit_result",
                        ["success"] = true,
                        ["save"] = saveResult,
                    };
                }
                sim.CleanUp();
                return new Dictionary<string, object?>
                {
                    ["type"] = "quit_result",
                    ["success"] = true,
                    ["save"] = null,
                };
            }

            default:
                return new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Unknown command: {cmdType}" };
        }
    }

    static void WriteLine(Dictionary<string, object?> data)
    {
        Console.Out.WriteLine(JsonSerializer.Serialize(data, JsonOpts));
        Console.Out.Flush();
    }
}
