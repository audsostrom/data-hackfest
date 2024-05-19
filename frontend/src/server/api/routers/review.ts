import { z } from 'zod';
import { eq } from 'drizzle-orm';
import { ilike } from "drizzle-orm";

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from '~/server/api/trpc';
import { reviews } from '~/server/db/schema';

//** movie route */
export const reviewRouter = createTRPCRouter({
  // defines purpose, protected = requires login
  create: protectedProcedure
  .input(z.object({ movieId: z.number(), rating: z.number(), desc: z.string().min(1) }))
  .mutation(async ({ ctx, input }) => {
    await ctx.db.insert(reviews).values({
      userId: ctx.session.user.id,
      movieId: input.movieId,
      rating: input.rating,
      desc: input.desc,
    });
  }),

  getXLatest: publicProcedure
  .input(z.number())
  .query(({ ctx, input }) => { 
    return ctx.db.query.reviews.findMany({
      orderBy: (reviews, { desc }) => [desc(reviews.id)],
      limit: input,
      with:{
        movie: true,
      }
    });
  }),

  getAll: protectedProcedure.query(({ ctx }) => {
    return ctx.db.query.reviews.findMany();
  }),
});
