/** 打包 live.js + pipeline.js + @johnhenry/laya → docs/assets/laya-live.bundle.js */
import * as esbuild from "esbuild";

await esbuild.build({
  entryPoints: ["live.js"],
  bundle: true,
  format: "esm",
  minify: true,
  target: ["es2022", "chrome110", "safari16"],
  outfile: "../docs/assets/laya-live.bundle.js",
  legalComments: "inline", // Apache NOTICE 随包保留
});
console.log("built -> docs/assets/laya-live.bundle.js");
