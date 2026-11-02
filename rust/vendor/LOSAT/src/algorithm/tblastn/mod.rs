//! TBLASTN algorithm module (protein query vs translated DNA subject)
//!
//! TBLASTN searches a protein query against a nucleotide subject translated
//! in all six reading frames. Implementation-wise it is the TBLASTX engine
//! with a single protein pseudo-frame per query (`query_is_protein`), so all
//! scanning, extension, HSP-linking, and statistics code is shared with
//! TBLASTX. The differences live at the edges:
//!
//! - the query FASTA holds protein residues (no query genetic code);
//! - query coordinates are reported in protein space (ascending, frame 0);
//! - the subject is translated with `--db-gencode`, like TBLASTX.
//!
//! NCBI scope note: this subcommand is experimental in v0.1.0, like BLASTN
//! and BLASTP. Parity evidence must accompany any release promotion.

use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

pub use super::tblastx;

/// Command-line arguments for TBLASTN.
///
/// Mirrors `TblastxArgs` minus the query genetic code (a protein query is not
/// translated) and converts into `TblastxArgs` with `query_is_protein = true`.
#[derive(Args, Debug)]
pub struct TblastnArgs {
    #[arg(short, long)]
    pub query: PathBuf,
    #[arg(short, long)]
    pub subject: PathBuf,
    #[arg(short, long, default_value_t = 10.0)]
    pub evalue: f64,
    #[arg(short, long, default_value_t = 13)]
    pub threshold: i32,
    #[arg(short = 'n', long, default_value_t = 0)]
    pub num_threads: usize,
    #[arg(short, long)]
    pub out: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    pub db_gencode: u8,
    #[arg(long, default_value_t = 500)]
    pub max_target_seqs: usize,
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub seg: bool,
    #[arg(long, default_value_t = 12)]
    pub seg_window: usize,
    #[arg(long, default_value_t = 2.2)]
    pub seg_locut: f64,
    #[arg(long, default_value_t = 2.5)]
    pub seg_hicut: f64,
    #[arg(long, default_value_t = 40)]
    pub window_size: usize,
    #[arg(long, default_value = "6")]
    pub outfmt: String,
    #[arg(long, default_value_t = 0)]
    pub culling_limit: u32,
}

impl From<TblastnArgs> for tblastx::TblastxArgs {
    fn from(args: TblastnArgs) -> Self {
        Self {
            query: args.query,
            subject: args.subject,
            evalue: args.evalue,
            threshold: args.threshold,
            word_size: 3,
            num_threads: args.num_threads,
            out: args.out,
            query_gencode: 1,
            db_gencode: args.db_gencode,
            max_target_seqs: args.max_target_seqs,
            seg: args.seg,
            seg_window: args.seg_window,
            seg_locut: args.seg_locut,
            seg_hicut: args.seg_hicut,
            window_size: args.window_size,
            outfmt: args.outfmt,
            culling_limit: args.culling_limit,
            query_is_protein: true,
        }
    }
}

/// Run a TBLASTN search: protein query vs translated nucleotide subject.
pub fn run(args: TblastnArgs) -> Result<()> {
    tblastx::run(args.into())
}
