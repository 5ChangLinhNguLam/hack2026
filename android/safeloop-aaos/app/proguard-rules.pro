# SourceMode.parse maps exact wire strings through Enum.valueOf. Preserve those
# three names in minified release builds.
-keep enum com.fptautomotive.safeloop.DecisionSnapshot$SourceMode { *; }
